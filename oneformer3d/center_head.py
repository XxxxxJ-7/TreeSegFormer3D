# Geometric center prediction branch for instance queries.
# Strategy: force each query to "know" the physical center of its instance,
# adding an explicit spatial distance loss to separate ambiguous instances.
#
# Two new registered modules (original code untouched):
#   1. ForAINetv2QueryDecoder_XAwarequery_Center   – decoder with center MLP
#   2. ForAINetV2OneFormer3D_XAwarequery_Center     – model with center loss
import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv
import MinkowskiEngine as ME
from torch_scatter import scatter_mean, scatter_add
from torch_cluster import fps

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector

from .query_decoder import ForAINetv2QueryDecoder_XAwarequery
from .oneformer3d import (
    ForAINetV2OneFormer3D_XAwarequery,
    _elastic_coords_as_tensor,
)
from .panoptic_losses import offset_loss, discriminative_loss, FastFocalLoss
from tools.base_modules import Seq, MLP


# ---------------------------------------------------------------------------
#  Decoder with center prediction head
# ---------------------------------------------------------------------------
@MODELS.register_module()
class ForAINetv2QueryDecoder_XAwarequery_Center(
        ForAINetv2QueryDecoder_XAwarequery):
    """Extends the XAwarequery decoder with a lightweight MLP that predicts
    the 3-D geometric center (x, y, z) for every instance query.

    Extra output key in the returned dict:
        ``centers``  – List[Tensor(n_queries, 3)] per batch element.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = kwargs['d_model']
        self.out_center = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),
        )

    # -- prediction head (adds center output) --------------------------------
    def _forward_head(self, queries, mask_feats):
        pred_scores, pred_masks, attn_masks = [], [], []
        pred_centers = []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            pred_score = self.out_score(norm_query) \
                if self.objectness_flag else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
            pred_centers.append(self.out_center(norm_query))
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks, pred_centers

    # -- simple (non-iterative) forward --------------------------------------
    def forward_simple(self, x, queries):
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        pred_scores, pred_masks, _, pred_centers = self._forward_head(
            queries, mask_feats)
        return dict(masks=pred_masks, scores=pred_scores,
                    centers=pred_centers)

    # -- iterative forward ---------------------------------------------------
    def forward_iter_pred(self, x, queries):
        pred_scores, pred_masks, pred_centers_all = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))

        pred_score, pred_mask, attn_mask, pred_center = \
            self._forward_head(queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        pred_centers_all.append(pred_center)

        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](
                inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask, pred_center = \
                self._forward_head(queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            pred_centers_all.append(pred_center)

        aux_outputs = [
            {'masks': masks, 'scores': scores, 'centers': ctrs}
            for scores, masks, ctrs in zip(
                pred_scores[:-1], pred_masks[:-1], pred_centers_all[:-1])]
        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            centers=pred_centers_all[-1],
            aux_outputs=aux_outputs)


# ---------------------------------------------------------------------------
#  Model with geometric-center loss
# ---------------------------------------------------------------------------
@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_Center(
        ForAINetV2OneFormer3D_XAwarequery):
    """Same as parent but adds a center regression loss (L1) that forces
    every positive instance query to predict the 3-D center of its GT
    instance.

    Extra config key:
        ``center_loss_weight`` (float): Weight of the center loss.
    """

    def __init__(self, center_loss_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.center_loss_weight = center_loss_weight

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _compute_gt_centers(voxel_coords, voxel_inst_labels):
        """Return a dict  {inst_id -> (3,) center}  from voxel data."""
        unique_ids = voxel_inst_labels.unique()
        centers = {}
        for uid in unique_ids:
            uid_val = uid.item()
            if uid_val < 0:
                continue
            mask = voxel_inst_labels == uid
            centers[uid_val] = voxel_coords[mask].mean(dim=0)
        return centers

    @staticmethod
    def _center_loss_for_sample(pred_centers, queries_inslabel,
                                gt_centers_dict):
        """L1 loss between predicted centres and GT centres.

        Args:
            pred_centers (Tensor): (n_queries, 3).
            queries_inslabel (Tensor): (n_queries,) – GT instance id per query
                (-1 for background / unassigned).
            gt_centers_dict (dict): {inst_id -> Tensor(3,)}.

        Returns:
            Tensor: scalar loss (0 if no valid pairs).
        """
        valid = queries_inslabel >= 0
        if not valid.any():
            return pred_centers.new_tensor(0.0)

        valid_idx = valid.nonzero(as_tuple=True)[0]
        pred = pred_centers[valid_idx]                          # (M, 3)
        gt_list = []
        keep = []
        for k, qi in enumerate(valid_idx):
            iid = queries_inslabel[qi].item()
            if iid in gt_centers_dict:
                gt_list.append(gt_centers_dict[iid])
                keep.append(k)
        if len(gt_list) == 0:
            return pred_centers.new_tensor(0.0)

        gt = torch.stack(gt_list)                               # (M', 3)
        pred = pred[keep]                                       # (M', 3)
        return F.l1_loss(pred, gt)

    # -- loss (overrides parent) ---------------------------------------------
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        # ====== Phase 1: backbone + discriminative / binary-semantic loss ====
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        embed_logits = [self.Embed(y) for y in x]
        bi_semantic_logits = [self.BiSemantic(y) for y in x]

        total_discriminative_loss = 0
        total_semantic_loss_bi = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            voxel_superpoints = inverse_mapping[
                coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]

            pts_instance_mask = \
                batch_data_samples[i].gt_pts_seg.pts_instance_mask
            instance_mask = \
                batch_data_samples[i].gt_pts_seg.instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            device = pts_instance_mask.device

            valid_instance_mask = instance_mask
            voxel_instance_labels = self.get_voxel_instance_labels(
                pts_instance_mask[valid_instance_mask],
                voxel_superpoints[valid_instance_mask])

            valid_voxel_indices = torch.unique(
                voxel_superpoints[valid_instance_mask])
            filtered_embed_logits = embed_logits[i][valid_voxel_indices]

            batch_idx = torch.full_like(voxel_instance_labels, i)
            disc_losses = discriminative_loss(
                filtered_embed_logits, voxel_instance_labels, batch_idx, 5)
            for ln, lv in disc_losses.items():
                total_discriminative_loss += lv if ln == "ins_loss" else 0

            bi_semantic_logit = bi_semantic_logits[i]
            instance_mask_tensor = torch.from_numpy(instance_mask)
            instance_mask = instance_mask_tensor.to(device)
            voxel_point_counts = scatter_add(
                torch.ones_like(instance_mask.float()),
                voxel_superpoints, dim=0)
            foreground_voxel_counts = scatter_add(
                instance_mask.float(), voxel_superpoints, dim=0)
            bi_y = (foreground_voxel_counts / voxel_point_counts) > 0.5
            bi_y = bi_y.long()
            semantic_loss_bi = F.nll_loss(bi_semantic_logit, bi_y)
            total_semantic_loss_bi += semantic_loss_bi

        total_discriminative_loss /= batch_size
        total_semantic_loss_bi /= batch_size

        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi,
        }

        # ====== Phase 2: decoder + instance criterion + center loss =========
        queries = []
        queries_inslabel = []
        queries_idx = []
        # Store per-sample voxel coords for center loss computation
        sample_voxel_coords = {}

        if self.prepare_epoch:
            cur_epoch = kwargs.get('epoch', self._train_epoch)
            if cur_epoch > self.prepare_epoch:

                for i in range(batch_size):
                    voxel_superpoints = inverse_mapping[
                        coordinates[:, 0][inverse_mapping] == i]
                    voxel_superpoints = torch.unique(
                        voxel_superpoints, return_inverse=True)[1]
                    pts_instance_mask = \
                        batch_data_samples[i].gt_pts_seg.pts_instance_mask
                    instance_mask = \
                        batch_data_samples[i].gt_pts_seg.instance_mask
                    valid_voxel_indices = torch.unique(
                        voxel_superpoints[instance_mask])

                    with torch.no_grad():
                        if valid_voxel_indices.numel() < 10:
                            queries.append([])
                            queries_inslabel.append([])
                            continue
                        voxel_instance_labels = \
                            self.get_voxel_instance_labels(
                                pts_instance_mask[instance_mask],
                                voxel_superpoints[instance_mask])

                        wood_class = 1
                        semantic_predictions_bi = torch.argmax(
                            bi_semantic_logits[i], dim=1)
                        tree_indices = torch.where(
                            semantic_predictions_bi == wood_class)[0]

                        batch_tensor = torch.zeros(
                            embed_logits[i][tree_indices].size(0),
                            dtype=torch.long,
                            device=embed_logits[i].device)
                        ratio = min(
                            self.query_point_num
                            / embed_logits[i][tree_indices].size(0),
                            torch.tensor([1.0],
                                         device=embed_logits[i].device))
                        topk_indices = fps(
                            embed_logits[i][tree_indices],
                            batch_tensor, ratio=ratio)
                        selected_indices = tree_indices[topk_indices]

                        # ---- voxel coords (needed for center loss) ----
                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(
                            current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()),
                            voxel_superpoints, dim=0)
                        avg_points = (
                            current_points_add
                            / voxel_counts.unsqueeze(-1).clamp(min=1))

                        queries.append(x[i][selected_indices])

                        pts_instance_mask_full = \
                            batch_data_samples[i].gt_pts_seg.pts_instance_mask
                        voxel_instance_labels_full = \
                            self.get_voxel_instance_labels(
                                pts_instance_mask_full, voxel_superpoints)

                        queries_inslabel.append(
                            voxel_instance_labels_full[selected_indices])
                        queries_idx.append(selected_indices)

                        sample_voxel_coords[i] = (
                            avg_points[:, :3],
                            voxel_instance_labels_full,
                        )

                if all(len(q) == 0 for q in queries):
                    pass
                else:
                    if any(len(q) == 0 for q in queries):
                        filtered = [
                            (x[i], queries[i], batch_data_samples[i],
                             queries_inslabel[i], i)
                            for i in range(len(queries))
                            if len(queries[i]) > 0
                        ]
                        x, queries, batch_data_samples, \
                            queries_inslabel, original_indices = \
                            zip(*filtered)
                        x = list(x)
                        queries = list(queries)
                        batch_data_samples = list(batch_data_samples)
                        queries_inslabel = list(queries_inslabel)
                        original_indices = list(original_indices)
                    else:
                        original_indices = list(range(len(batch_data_samples)))

                    decoder_out = self.decoder(x, queries)

                    sp_gt_instances = []
                    for i in range(len(batch_data_samples)):
                        voxel_superpoints = inverse_mapping[
                            coordinates[:, 0][inverse_mapping]
                            == original_indices[i]]
                        voxel_superpoints = torch.unique(
                            voxel_superpoints, return_inverse=True)[1]
                        inst_mask = \
                            batch_data_samples[i].gt_pts_seg.pts_instance_mask
                        sem_mask = \
                            batch_data_samples[i].gt_pts_seg.pts_semantic_mask
                        assert voxel_superpoints.shape == inst_mask.shape

                        bd = batch_data_samples[i]
                        bd.gt_instances_3d.sp_sem_masks = \
                            self.get_gt_semantic_masks(
                                sem_mask, voxel_superpoints, self.num_classes)
                        bd.gt_instances_3d.sp_inst_masks = \
                            self.get_gt_inst_masks(
                                inst_mask, voxel_superpoints)
                        (bd.gt_instances_3d.labels_3d,
                         bd.gt_instances_3d.sp_inst_masks,
                         bd.gt_instances_3d.ratio_inspoint) = \
                            self.filter_stuff_masks(
                                bd.gt_instances_3d, self.stuff_classes,
                                bd.gt_pts_seg.ratio_inspoint)
                        bd.gt_instances_3d.query_inslabel = \
                            queries_inslabel[i]

                        sp_gt_instances.append(bd.gt_instances_3d)

                    loss = self.criterion(decoder_out, sp_gt_instances)
                    loss_final.update(loss)

                    # ---- center loss (new) ----
                    if 'centers' in decoder_out:
                        center_losses = []
                        pred_centers_list = decoder_out['centers']
                        for i in range(len(original_indices)):
                            oi = original_indices[i]
                            if oi not in sample_voxel_coords:
                                continue
                            vox_coords, vox_inst_labels = \
                                sample_voxel_coords[oi]
                            gt_ctr = self._compute_gt_centers(
                                vox_coords, vox_inst_labels)
                            # strip semantic queries (last n)
                            n_sem = self.num_classes
                            inst_centers = pred_centers_list[i][:-n_sem]
                            cl = self._center_loss_for_sample(
                                inst_centers, queries_inslabel[i], gt_ctr)
                            center_losses.append(cl)

                        # aux layers
                        if 'aux_outputs' in decoder_out:
                            for aux in decoder_out['aux_outputs']:
                                if 'centers' not in aux:
                                    continue
                                aux_ctrs = aux['centers']
                                for i in range(len(original_indices)):
                                    oi = original_indices[i]
                                    if oi not in sample_voxel_coords:
                                        continue
                                    vox_coords, vox_inst_labels = \
                                        sample_voxel_coords[oi]
                                    gt_ctr = self._compute_gt_centers(
                                        vox_coords, vox_inst_labels)
                                    n_sem = self.num_classes
                                    inst_ctrs = aux_ctrs[i][:-n_sem]
                                    cl = self._center_loss_for_sample(
                                        inst_ctrs,
                                        queries_inslabel[i], gt_ctr)
                                    center_losses.append(cl)

                        if center_losses:
                            loss_final['center_loss'] = (
                                self.center_loss_weight
                                * torch.stack(center_losses).mean())

        return loss_final
