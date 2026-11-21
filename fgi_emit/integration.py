"""MMDetection3D adapters; existing FOR-instance/Xiongan models are unchanged."""
import json
from pathlib import Path
import subprocess
import sys

import laspy
import numpy as np
from scipy.spatial import cKDTree
import torch
import spconv.pytorch as spconv
from torch_cluster import fps
from mmcv.transforms import BaseTransform
from mmengine.dist import get_world_size
from mmengine.evaluator import BaseMetric
from mmdet3d.datasets.scannet_dataset import ScanNetDataset
from mmdet3d.registry import DATASETS, METRICS, MODELS, TRANSFORMS
from mmdet3d.structures import PointData

import oneformer3d  # Register the original backbones, losses and decoders.
from oneformer3d.oneformer3d import ForAINetV2OneFormer3D, ForAINetV2OneFormer3D_XAwarequery
from oneformer3d.ias_bcl_v3 import ForAINetV2OneFormer3D_XAwarequery_IAS_BCL_V3
from oneformer3d.mask_matrix_nms import mask_matrix_nms
from .protocol import InstanceMerger, SPLITS, tile_centers, voxel_representatives


@DATASETS.register_module()
class FGIEMITDataset(ScanNetDataset):
    METAINFO = dict(classes=('other', 'tree'), palette=[[128, 128, 128], [34, 139, 34]],
                    seg_valid_class_ids=(0, 1), seg_all_class_ids=(0, 1))

    def get_scene_idxs(self, *args, **kwargs):
        return np.arange(len(self), dtype=np.int32)


@TRANSFORMS.register_module()
class FGIEMITTrainSample(BaseTransform):
    """Crop/sample once, then remap instances and their completeness ratios together."""
    def __init__(self, radius=12.0, voxel_size=0.2, max_points=640000):
        self.radius, self.voxel_size, self.max_points = radius, voxel_size, max_points

    def transform(self, results):
        points = results['points']
        xyz = points.tensor.numpy()[:, :3]
        original_inst = np.asarray(results['pts_instance_mask'])
        if not len(xyz):
            return None
        center = xyz[np.random.randint(len(xyz)), :2]
        crop = np.flatnonzero(((xyz[:, :2] - center) ** 2).sum(1) <= self.radius ** 2)
        # Random representative per voxel; no hash collision and no replacement.
        shuffled = np.random.permutation(crop)
        reps, _ = voxel_representatives(xyz[shuffled], self.voxel_size)
        chosen = shuffled[reps]
        if len(chosen) > self.max_points:
            chosen = np.random.choice(chosen, self.max_points, replace=False)
        ids = np.unique(original_inst[chosen])
        ids = ids[ids > 0]
        if not len(ids):
            return None
        remapped = np.full(len(chosen), -1, dtype=np.int64)
        ratios = {}
        votes = np.zeros((len(chosen), 3), dtype=np.float32)
        for new_id, old_id in enumerate(ids):
            full_mask = original_inst == old_id
            selected_mask = original_inst[chosen] == old_id
            remapped[selected_mask] = new_id
            ratios[new_id] = float(np.count_nonzero(original_inst[crop] == old_id) / full_mask.sum())
            instance_xyz = xyz[full_mask]
            middle = (instance_xyz.max(0) + instance_xyz.min(0)) / 2
            votes[selected_mask] = middle - xyz[chosen[selected_mask]]
        results.update(points=points[chosen], pts_instance_mask=remapped,
                       pts_semantic_mask=np.asarray(results['pts_semantic_mask'])[chosen],
                       gt_labels_3d=np.ones(len(ids), dtype=np.int64),
                       ratio_inspoint=ratios, vote_label=torch.from_numpy(votes),
                       instance_mask=remapped >= 0)
        return results


@TRANSFORMS.register_module()
class FGIEMITMakeWritable(BaseTransform):
    """Copy read-only arrays produced by ``np.frombuffer`` before packing."""
    def transform(self, results):
        for key in ('pts_instance_mask', 'pts_semantic_mask'):
            value = results.get(key)
            if isinstance(value, np.ndarray) and not value.flags.writeable:
                results[key] = value.copy()
        if 'eval_ann_info' in results:
            for key in ('pts_instance_mask', 'pts_semantic_mask'):
                value = results['eval_ann_info'].get(key)
                if isinstance(value, np.ndarray) and not value.flags.writeable:
                    results['eval_ann_info'][key] = value.copy()
        return results


class FGIEMITInferenceMixin:
    """Common full-plot tiling, preserving each model's decoder/query selection.

    Inference reads only XYZ, never eval_ann_info. The full point mapping is
    restored before evaluation. This is a controlled XYZ adaptation, not a
    reproduction of every model's original full-plot postprocessing recipe.
    """
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        # BCL's original implementation expects epoch via a patched runner.
        # SetTrainEpochHook provides it without editing installed MMEngine files.
        kwargs.setdefault('epoch', getattr(self, '_train_epoch', 0))
        return super().loss(batch_inputs_dict, batch_data_samples, **kwargs)

    def _fgi_tile(self, points, tile_center=None):
        coordinates, features, inverse, shape = self.collate([points])
        sparse = spconv.SparseConvTensor(features, coordinates, shape, 1)
        feats = self.extract_feat(sparse)
        if hasattr(self, 'Embed'):
            embedding = self.Embed(feats[0])
            foreground = torch.where(self.BiSemantic(feats[0]).argmax(1) == 1)[0]
            if foreground.numel() <= 1:
                indices = torch.arange(min(self.query_point_num, len(feats[0])), device=points.device)
            elif hasattr(self, '_select_query_indices'):
                indices = self._select_query_indices(embedding, foreground, points.device)
            else:
                local = fps(embedding[foreground],
                            torch.zeros(len(foreground), dtype=torch.long, device=points.device),
                            ratio=min(1.0, self.query_point_num / len(foreground)),
                            random_start=False)
                indices = foreground[local[:self.query_point_num]]
            out = self.decoder(feats, [feats[0][indices]])
        else:
            out = self.decoder(feats)
        # Shared, dynamic-query-safe decoding; semantic queries are the final two.
        logits = out['masks'][0]
        semantic = logits[-2:, :].argmax(0)[inverse]
        masks = logits[:-2, :].sigmoid()
        scores = out['scores'][0][:-2].flatten()
        scores, order = scores.topk(min(len(scores), self.test_cfg.topk_insts), sorted=True)
        masks = masks[order]
        if len(scores):
            valid = masks > self.test_cfg.obj_normalization_thr
            scores = scores * (masks * valid).sum(1) / (valid.sum(1) + 1e-6)
            scores, _, masks, _ = mask_matrix_nms(
                masks, torch.zeros_like(scores, dtype=torch.long), scores,
                kernel=self.test_cfg.matrix_nms_kernel)
        masks = masks[:, inverse] > self.test_cfg.sp_score_thr
        # Match the existing foreground-majority rule before shared tile merging.
        count = masks.sum(1)
        keep = ((scores > self.test_cfg.inst_score_thr) & torch.isfinite(scores)
                & (count > self.test_cfg.npoint_thr)
                & ((masks & (semantic == 0)).sum(1) <= count / 2))
        # ForestFormer3D's released full-plot postprocessor rejects masks that
        # touch the cylinder boundary. With overlapping 12 m cylinders this
        # keeps complete interior predictions and suppresses clipped crowns.
        edge_margin = float(self.test_cfg.get('edge_margin', 0.0))
        if tile_center is not None and edge_margin > 0 and len(masks):
            center = torch.as_tensor(
                tile_center, dtype=points.dtype, device=points.device)
            distance = torch.linalg.vector_norm(points[:, :2] - center, dim=1)
            near_edge = distance > (float(self.radius) - edge_margin)
            keep &= ~(masks & near_edge.unsqueeze(0)).any(1)
        return semantic.cpu().numpy(), masks[keep].cpu().numpy(), scores[keep].cpu().numpy()

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        if len(batch_data_samples) != 1:
            raise ValueError('FGI-EMIT full-plot inference requires batch_size=1.')
        points = batch_inputs_dict['points'][0]
        xyz = points[:, :3].detach().cpu().numpy()
        representative, inverse = voxel_representatives(xyz, self.voxel_size)
        sparse_xyz = xyz[representative]
        lookup = cKDTree(sparse_xyz[:, :2])
        votes = np.zeros((len(sparse_xyz), 2), dtype=np.int32)
        merger = InstanceMerger(len(sparse_xyz), self.test_cfg.merge_iou)
        plot = int(batch_data_samples[0].plot_id)
        seed = int(self.test_cfg.inference_seed) + plot
        rng = np.random.default_rng(seed)
        # Isolate inference RNG from training RNG, including IAS's FPS/random budget.
        devices = ([points.device.index if points.device.index is not None
                    else torch.cuda.current_device()] if points.is_cuda else [])
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for center in tile_centers(sparse_xyz, self.radius, self.test_cfg.tile_stride):
                tile = np.asarray(sorted(lookup.query_ball_point(center, self.radius)), dtype=np.int64)
                if not len(tile):
                    continue
                selected = tile
                mapping = np.arange(len(tile))
                if len(tile) > self.test_cfg.max_points:
                    selected = np.sort(rng.choice(tile, self.test_cfg.max_points, replace=False))
                    _, mapping = cKDTree(sparse_xyz[selected]).query(sparse_xyz[tile])
                tensor_idx = torch.as_tensor(representative[selected], device=points.device)
                semantic, masks, scores = self._fgi_tile(
                    points[tensor_idx], tile_center=center)
                semantic = semantic[mapping]
                votes[tile, semantic] += 1
                for index in np.argsort(-scores, kind='stable'):
                    merger.add(tile, masks[index][mapping], float(scores[index]))
        if not (votes.sum(1) > 0).all():
            raise RuntimeError('Tiling left points unevaluated.')
        semantic = votes.argmax(1)
        labels, confidence = merger.finish(semantic)
        batch_data_samples[0].pred_pts_seg = PointData(
            pts_semantic_mask=semantic[inverse], pts_instance_mask=labels[inverse],
            point_confidence=confidence[inverse])
        return batch_data_samples


@MODELS.register_module()
class FGIEMITOneFormer3D(FGIEMITInferenceMixin, ForAINetV2OneFormer3D):
    pass


@MODELS.register_module()
class FGIEMITForestFormer3D(FGIEMITInferenceMixin, ForAINetV2OneFormer3D_XAwarequery):
    pass


@MODELS.register_module()
class FGIEMITTreeSegFormer3D(FGIEMITInferenceMixin, ForAINetV2OneFormer3D_XAwarequery_IAS_BCL_V3):
    pass


@METRICS.register_module()
class FGIEMITOfficialMetric(BaseMetric):
    default_prefix = 'fgi_emit'

    def __init__(self, split, pred_root, data_root='data/FGI_EMIT/processed',
                 raw_root='data/FGI_EMIT/raw', **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.pred_root, self.data_root, self.raw_root = map(Path, (pred_root, data_root, raw_root))

    def process(self, data_batch, data_samples):
        if get_world_size() != 1:
            raise RuntimeError('Use the documented single-GPU benchmark (no distributed LAS writes).')
        for sample in data_samples:
            # MMEngine's Evaluator converts BaseDataElement objects to dicts
            # before calling metrics, while direct metric use may retain the
            # original object. Support both representations.
            get_field = sample.get if isinstance(sample, dict) else \
                lambda name: getattr(sample, name)
            ident = int(get_field('plot_id'))
            stem = f'plot_{ident}'
            xyz = np.fromfile(self.data_root / 'points' / f'{stem}.bin', dtype='<f4').reshape(-1, 3)
            pred = get_field('pred_pts_seg')
            semantic = np.asarray(pred['pts_semantic_mask'])
            instance = np.asarray(pred['pts_instance_mask'])
            confidence = np.asarray(pred['point_confidence'])
            if not (semantic.shape == instance.shape == confidence.shape == (len(xyz),)):
                raise ValueError('Prediction length does not match the complete plot.')
            gt_sem = np.fromfile(self.data_root / 'semantic_mask' / f'{stem}.bin', dtype='<i8')
            gt_inst = np.fromfile(self.data_root / 'instance_mask' / f'{stem}.bin', dtype='<i8')
            header = laspy.LasHeader(point_format=3, version='1.2')
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.asarray(get_field('xyz_offset'), dtype=np.float64)
            for name, dtype in [('tree_index', 'u4'), ('tree_pred', 'u4'),
                                ('semantic_pred', 'u1'), ('score', 'f4')]:
                header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
            cloud = laspy.LasData(header)
            cloud.xyz = xyz.astype(np.float64) + header.offsets
            cloud.classification = gt_sem.astype(np.uint8)
            cloud['tree_index'] = gt_inst.astype(np.uint32)
            cloud['tree_pred'] = instance.astype(np.uint32)
            cloud['semantic_pred'] = semantic.astype(np.uint8)
            cloud['score'] = confidence.astype(np.float32)
            dest = self.pred_root / self.split / f'{stem}.las'
            dest.parent.mkdir(parents=True, exist_ok=True)
            cloud.write(dest)
            self.results.append(ident)

    def compute_metrics(self, results):
        if sorted(results) != sorted(SPLITS[self.split]):
            raise ValueError(f'Expected complete {self.split} set; received {results}')
        output = self.pred_root / f'{self.split}_metrics.json'
        subprocess.run([sys.executable, '-m', 'fgi_emit.evaluate',
                        '--pred-root', str(self.pred_root.resolve()),
                        '--raw-root', str(self.raw_root.resolve()), '--split', self.split,
                        '--output', str(output.resolve())], check=True,
                       cwd=Path(__file__).resolve().parents[1])
        return json.loads(output.read_text(encoding='utf-8'))
