"""Multi-Scale Cross-Attention for ForestFormer3D Decoder.

Introduces multi-scale encoder features into the Decoder's Cross-Attention,
where each Decoder layer has multiple Cross-Attention heads, each focusing
on encoder features from a different U-Net stage (Stage1/Stage2/Stage3).
Outputs from all scales are fused via concatenation+projection.

New components:
    MultiScaleCrossAttentionLayer  – multi-head cross-attn over multiple scales
    ForAINetv2QueryDecoder_XAwarequery_MultiScale  – decoder with multi-scale CA
    ForAINetV2OneFormer3D_XAwarequery_MultiScale   – model with multi-scale feat extraction
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
import os
import MinkowskiEngine as ME

from tqdm import tqdm
from torch_scatter import scatter_add
from torch_cluster import fps

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from .query_decoder import (
    SelfAttentionLayer, FFN, ForAINetv2QueryDecoder_XAwarequery
)
from .oneformer3d import ForAINetV2OneFormer3D_XAwarequery
from .panoptic_losses import discriminative_loss
import contextlib, time

# =========================================================================
#  MultiScaleCrossAttentionLayer
# =========================================================================

class MultiScaleCrossAttentionLayer(BaseModule):
    """Multi-scale cross-attention: one cross-attention head per encoder scale.

    Instead of a single cross-attention over the final encoder features,
    this layer has multiple parallel cross-attention heads, each attending
    to features from a different U-Net stage.  The outputs are fused via
    concatenation + linear projection (default) or a learnable weighted sum.

    Args:
        d_model (int): Model / embedding dimension.
        num_heads (int): Number of attention heads per scale.
        dropout (float): Attention dropout rate.
        num_scales (int): How many encoder scales to attend to (default 3).
        fusion_mode (str): 'concat' (concat → Linear) or 'weighted_sum'.
        fix (bool): Enable residual dropout & LayerNorm (same as fix_attention).
    """

    def __init__(self, d_model, num_heads, dropout, num_scales=3,
                 fusion_mode='concat', fix=False):
        super().__init__()
        self.fix = fix
        self.num_scales = num_scales
        self.fusion_mode = fusion_mode

        # ---- per-scale cross-attention heads ----
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads,
                                  dropout=dropout, batch_first=True)
            for _ in range(num_scales)
        ])

        # ---- fusion after collecting all scales ----
        if fusion_mode == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(d_model * num_scales, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout) if fix else nn.Identity(),
            )
        elif fusion_mode == 'weighted_sum':
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if fix else nn.Identity()
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, multi_sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            multi_sources: ``List[List[Tensor]]`` shape
                ``[num_scales][batch_size][n_points_s, d_model]``.
            queries: ``List[Tensor]`` shape
                ``[batch_size][n_queries, d_model]``.
            attn_masks: Optional ``List[Tensor]``, shape
                ``[batch_size][n_queries, n_points_finest]``.
                Applied only to the *finest* scale (scale 0) because
                coarser scales have different point counts.

        Returns:
            List[Tensor]: shape ``[batch_size][n_queries, d_model]``.
        """
        outputs = []
        for i in range(len(queries)):
            scale_outputs = []
            for s in range(self.num_scales):
                k = v = multi_sources[s][i]
                # attention mask only for the finest scale
                mask = attn_masks[i] if (attn_masks is not None and s == 0) else None
                out, _ = self.attns[s](queries[i], k, v, attn_mask=mask)
                scale_outputs.append(out)

            # ---- fuse per-scale outputs ----
            if self.fusion_mode == 'concat':
                fused = torch.cat(scale_outputs, dim=-1)
                fused = self.fusion(fused)
            elif self.fusion_mode == 'weighted_sum':
                w = torch.softmax(self.scale_weights, dim=0)
                fused = sum(wgt * s for wgt, s in zip(w, scale_outputs))

            # ---- residual + norm (Transformer post-norm pattern) ----
            if self.fix:
                fused = self.dropout(fused)
            fused = fused + queries[i]
            if self.fix:
                fused = self.norm(fused)
            outputs.append(fused)

        return outputs


# =========================================================================
#  Multi-Scale Query Decoder
# =========================================================================

@MODELS.register_module()
class ForAINetv2QueryDecoder_XAwarequery_MultiScale(BaseModule):
    """Query decoder with multi-scale cross-attention.

    Each decoder layer's cross-attention attends to encoder features from
    multiple U-Net stages simultaneously, fusing the outputs.

    Args:
        num_layers (int): Number of transformer decoder layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Feature dimension of the FINEST scale.
        mid_channels (tuple[int]): Feature dimensions of coarser scales,
            e.g. (64, 96) for 2 additional scales.
        d_model (int): Transformer hidden dimension.
        num_heads (int): Number of attention heads per scale.
        hidden_dim (int): FFN hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
        iter_pred (bool): Whether to use iterative mask prediction.
        attn_mask (bool): Whether to use mask-guided attention.
        fix_attention (bool): Fix attention pattern.
        objectness_flag (bool): Whether to predict objectness score.
        num_scales (int): Total number of encoder scales (default 3).
        fusion_mode (str): How to fuse multi-scale outputs ('concat' or 'weighted_sum').
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, mid_channels=(64, 96), d_model=256,
                 num_heads=8, hidden_dim=1024, dropout=0.0,
                 activation_fn='gelu', iter_pred=True, attn_mask=True,
                 fix_attention=True, objectness_flag=True,
                 num_scales=3, fusion_mode='concat', **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.num_scales = num_scales

        # ---- per-scale input projections ----
        all_in_channels = [in_channels] + list(mid_channels[:num_scales - 1])
        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ch, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU())
            for ch in all_in_channels
        ])

        # --- query construction (unchanged from original) ---
        if num_instance_queries > 0:
            self.query = nn.Embedding(
                num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
            self.num_semantic_queries = num_instance_queries + num_semantic_queries
            self.semantic_queries = nn.Embedding(num_semantic_queries, in_channels)

        # ---- transformer layers ----
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                MultiScaleCrossAttentionLayer(
                    d_model, num_heads, dropout,
                    num_scales=num_scales, fusion_mode=fusion_mode,
                    fix=fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))

        # ---- output heads ----
        self.out_norm = nn.LayerNorm(d_model)
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))

    # ------------------------------------------------------------------
    #  Query construction  (unchanged from the original)
    # ------------------------------------------------------------------

    def _get_queries(self, queries=None, batch_size=None):
        if batch_size is None:
            batch_size = len(queries)
        result_queries = []
        for i in range(batch_size):
            if len(queries[i]) != 0:
                device = queries[i].device
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                semantic_queries = self.semantic_queries.weight
                if len(queries[i]) == 0:
                    queries[i] = torch.empty(
                        0, *semantic_queries.shape[1:]).to(device)
                concat_queries = torch.cat(
                    (queries[i], semantic_queries.to(queries[i].device)), dim=0)
                result_query.append(self.query_proj(concat_queries))
            result_queries.append(torch.cat(result_query))
        return result_queries

    # ------------------------------------------------------------------
    #  Prediction head  (unchanged)
    # ------------------------------------------------------------------

    def _forward_head(self, queries, mask_feats):
        pred_scores, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return pred_scores, pred_masks, attn_masks

    # ------------------------------------------------------------------
    #  Simple (non-iterative) forward
    # ------------------------------------------------------------------

    def forward_simple(self, x, queries):
        """Multi-scale simple forward.

        Args:
            x (tuple): ``(fine_feats, mid1_feats, mid2_feats, ...)`` where
                each is ``List[Tensor]`` of len batch_size.
            queries: content queries.
        """
        # project each scale to d_model
        multi_inst_feats = [
            [self.input_projs[s](y) for y in x[s]]
            for s in range(self.num_scales)
        ]
        mask_feats = [self.x_mask(y) for y in x[0]]
        queries = self._get_queries(queries, len(x[0]))

        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](multi_inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)

        pred_scores, pred_masks, _ = self._forward_head(queries, mask_feats)
        return dict(masks=pred_masks, scores=pred_scores)

    # ------------------------------------------------------------------
    #  Iterative prediction forward  (the main path)
    # ------------------------------------------------------------------

    def forward_iter_pred(self, x, queries):
        """Multi-scale iterative forward pass.

        Args:
            x (tuple): ``(fine_feats, mid1_feats, mid2_feats, ...)`` where
                each is ``List[Tensor]`` of len batch_size.
            queries: content queries (List[Tensor]).
        """
        pred_scores, pred_masks = [], []

        # project each scale to d_model
        multi_inst_feats = [
            [self.input_projs[s](y) for y in x[s]]
            for s in range(self.num_scales)
        ]
        mask_feats = [self.x_mask(y) for y in x[0]]
        queries = self._get_queries(queries, len(x[0]))

        pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](
                multi_inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'masks': masks, 'scores': scores}
            for scores, masks in zip(pred_scores[:-1], pred_masks[:-1])]
        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    # ------------------------------------------------------------------
    #  Public forward
    # ------------------------------------------------------------------

    def forward(self, x, queries=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


# =========================================================================
#  Multi-Scale Model
# =========================================================================

@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_MultiScale(
        ForAINetV2OneFormer3D_XAwarequery):
    """ForAINetV2OneFormer3D_XAwarequery with multi-scale decoder cross-attn.

    Differences from the base class:
    1. ``extract_feat`` now extracts encoder features from **three** U-Net
       stages (fine / mid1 / mid2) and returns them as a tuple.
    2. The decoder is ``ForAINetv2QueryDecoder_XAwarequery_MultiScale``
       which uses ``MultiScaleCrossAttentionLayer`` in each decoder layer.

    Config usage::

        model = dict(
            type='ForAINetV2OneFormer3D_XAwarequery_MultiScale',
            backbone=dict(
                type='SpConvUNet',
                num_planes=[32, 64, 96, 128, 160],
                return_blocks=True),          # <-- must be True
            decoder=dict(
                type='ForAINetv2QueryDecoder_XAwarequery_MultiScale',
                mid_channels=(64, 96),        # channels of the two extra scales
                num_scales=3,
                fusion_mode='concat',         # 'concat' or 'weighted_sum'
                ...),
            ...)
    """

    def extract_feat(self, x):
        """Extract multi-scale features from the 3D sparse U-Net.

        Returns:
            tuple: ``(fine_list, mid1_list, mid2_list)`` where each is
            ``List[Tensor]`` of len batch_size.
            - fine_list  : 32-ch features at original resolution  (Stage3)
            - mid1_list  : 64-ch features at stride-2 resolution (Stage2)
            - mid2_list  : 96-ch features at stride-4 resolution (Stage1)
        """
        x = self.input_conv(x)
        output, blocks = self.unet(x)          # blocks: [level4..level0]
        x = self.output_layer(output)

        # ---- fine scale (32 ch, original resolution) ----
        fine_list = []
        for i in x.indices[:, 0].unique():
            fine_list.append(x.features[x.indices[:, 0] == i])

        # ---- mid scale 1 (64 ch, stride-2) ----
        mid1_tensor = blocks[-2]               # level 1 features
        mid1_list = []
        for i in mid1_tensor.indices[:, 0].unique():
            mid1_list.append(
                mid1_tensor.features[mid1_tensor.indices[:, 0] == i])

        # ---- mid scale 2 (96 ch, stride-4) ----
        mid2_tensor = blocks[-3]               # level 2 features
        mid2_list = []
        for i in mid2_tensor.indices[:, 0].unique():
            mid2_list.append(
                mid2_tensor.features[mid2_tensor.indices[:, 0] == i])

        return fine_list, mid1_list, mid2_list

    # ------------------------------------------------------------------
    #  loss  – override to unpack multi-scale features
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses (multi-scale variant)."""
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        # ---- unpack multi-scale features ----
        fine_feats, mid1_feats, mid2_feats = self.extract_feat(x)

        embed_logits = [self.Embed(y) for y in fine_feats]
        bi_semantic_logits = [self.BiSemantic(y) for y in fine_feats]

        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bi = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][
                inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]

            pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask

            device = pts_instance_mask.device
            valid_instance_mask = instance_mask

            voxel_instance_labels = self.get_voxel_instance_labels(
                pts_instance_mask[valid_instance_mask],
                voxel_superpoints[valid_instance_mask])

            valid_voxel_indices = torch.unique(
                voxel_superpoints[valid_instance_mask])

            filtered_embed_logits = embed_logits[i][valid_voxel_indices]

            batch_v = torch.full_like(voxel_instance_labels, i)
            discriminative_losses = discriminative_loss(
                filtered_embed_logits, voxel_instance_labels, batch_v, 5)

            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += (
                    loss if loss_name == "ins_loss" else 0)

            bi_semantic_logit = bi_semantic_logits[i]
            instance_mask_tensor = torch.from_numpy(instance_mask)
            instance_mask_dev = instance_mask_tensor.to(device)
            voxel_point_counts = scatter_add(
                torch.ones_like(instance_mask_dev.float()),
                voxel_superpoints, dim=0)
            foreground_voxel_counts = scatter_add(
                instance_mask_dev.float(), voxel_superpoints, dim=0)
            bi_y = (foreground_voxel_counts / voxel_point_counts) > 0.5
            bi_y = bi_y.long()
            semantic_loss_bi = torch.nn.functional.nll_loss(
                bi_semantic_logit, bi_y.to(torch.int64))
            total_semantic_loss_bi += semantic_loss_bi

        total_discriminative_loss /= batch_size
        total_semantic_loss_bi /= batch_size

        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi}

        # ---- query construction & decoder (same as original) ----
        queries = []
        queries_inslabel = []
        queries_idx = []

        if self.prepare_epoch:
            cur_epoch = kwargs.get('epoch', self._train_epoch)
            if cur_epoch > self.prepare_epoch:
                total_qscore_loss = 0
                for i in range(batch_size):
                    voxel_superpoints = inverse_mapping[coordinates[:, 0][
                        inverse_mapping] == i]
                    voxel_superpoints = torch.unique(
                        voxel_superpoints, return_inverse=True)[1]
                    pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
                    instance_mask = batch_data_samples[i].gt_pts_seg.instance_mask
                    valid_voxel_indices = torch.unique(
                        voxel_superpoints[instance_mask])

                    with torch.no_grad():
                        if valid_voxel_indices.numel() < 10:
                            queries.append([])
                            queries_inslabel.append([])
                            continue
                        voxel_instance_labels = self.get_voxel_instance_labels(
                            pts_instance_mask[instance_mask],
                            voxel_superpoints[instance_mask])

                        wood_class = 1
                        semantic_predictions_bi = torch.argmax(
                            bi_semantic_logits[i], dim=1)
                        tree_indices = torch.where(
                            semantic_predictions_bi == wood_class)[0]

                        batch_tensor_4 = torch.zeros(
                            embed_logits[i][tree_indices].size(0),
                            dtype=torch.long).to(embed_logits[i].device)
                        topk_indices_4 = fps(
                            embed_logits[i][tree_indices], batch_tensor_4,
                            ratio=min(
                                self.query_point_num /
                                embed_logits[i][tree_indices].size(0),
                                torch.tensor([1.0]).to(embed_logits[i].device)))
                        selected_indices_case4 = tree_indices[topk_indices_4]

                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(
                            current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()),
                            voxel_superpoints, dim=0)
                        avg_points = current_points_add / voxel_counts.unsqueeze(
                            -1).clamp(min=1)

                        queries.append(fine_feats[i][selected_indices_case4])

                        pts_instance_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
                        voxel_instance_labels = self.get_voxel_instance_labels(
                            pts_instance_mask, voxel_superpoints)
                        queries_inslabel.append(
                            voxel_instance_labels[selected_indices_case4])
                        queries_idx.append(selected_indices_case4)

                if all(len(q) == 0 for q in queries):
                    pass
                else:
                    if any(len(q) == 0 for q in queries):
                        filtered_results = [
                            (fine_feats[i], queries[i],
                             batch_data_samples[i], queries_inslabel[i], i)
                            for i in range(len(queries))
                            if len(queries[i]) > 0]
                        fine_feats_f, queries_f, batch_data_samples_f, queries_inslabel_f, original_indices = \
                            zip(*filtered_results)
                        fine_feats_f = list(fine_feats_f)
                        queries_f = list(queries_f)
                        batch_data_samples_f = list(batch_data_samples_f)
                        queries_inslabel_f = list(queries_inslabel_f)
                        original_indices = list(original_indices)

                        # filter mid features to match
                        mid1_feats_f = [mid1_feats[j] for j in original_indices]
                        mid2_feats_f = [mid2_feats[j] for j in original_indices]
                    else:
                        fine_feats_f = fine_feats
                        queries_f = queries
                        batch_data_samples_f = batch_data_samples
                        queries_inslabel_f = queries_inslabel
                        original_indices = list(range(len(batch_data_samples)))
                        mid1_feats_f = mid1_feats
                        mid2_feats_f = mid2_feats

                    # ---- KEY CHANGE: pass multi-scale features to decoder ----
                    ms_feats = (fine_feats_f, mid1_feats_f, mid2_feats_f)
                    x_dec = self.decoder(ms_feats, queries_f)

                    sp_gt_instances = []
                    for i in range(len(batch_data_samples_f)):
                        voxel_superpoints = inverse_mapping[coordinates[:, 0][
                            inverse_mapping] == original_indices[i]]
                        voxel_superpoints = torch.unique(
                            voxel_superpoints, return_inverse=True)[1]
                        inst_mask = batch_data_samples_f[i].gt_pts_seg.pts_instance_mask
                        sem_mask = batch_data_samples_f[i].gt_pts_seg.pts_semantic_mask
                        assert voxel_superpoints.shape == inst_mask.shape

                        batch_data_samples_f[i].gt_instances_3d.sp_sem_masks = \
                            self.get_gt_semantic_masks(
                                sem_mask, voxel_superpoints, self.num_classes)
                        batch_data_samples_f[i].gt_instances_3d.sp_inst_masks = \
                            self.get_gt_inst_masks(inst_mask, voxel_superpoints)

                        (batch_data_samples_f[i].gt_instances_3d.labels_3d,
                         batch_data_samples_f[i].gt_instances_3d.sp_inst_masks,
                         batch_data_samples_f[i].gt_instances_3d.ratio_inspoint) = \
                            self.filter_stuff_masks(
                                batch_data_samples_f[i].gt_instances_3d,
                                self.stuff_classes,
                                batch_data_samples_f[i].gt_pts_seg.ratio_inspoint)

                        batch_data_samples_f[i].gt_instances_3d.query_inslabel = \
                            queries_inslabel_f[i]
                        sp_gt_instances.append(
                            batch_data_samples_f[i].gt_instances_3d)

                    loss = self.criterion(x_dec, sp_gt_instances)
                    loss_final.update(loss)
        return loss_final

    # ------------------------------------------------------------------
    #  predict  – override to unpack multi-scale features
    # ------------------------------------------------------------------

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
    #def predict_rm_outputpoints(self, batch_inputs_dict, batch_data_samples, **kwargs):
        t0 = time.time()            
        lidar_path = batch_data_samples[0].lidar_path
        base_name = os.path.basename(lidar_path)
        current_filename = os.path.splitext(base_name)[0]
        t1 = time.time()                 
        #########print(f"load pc: {(t1 - t0)*1000:.0f} ms")
        #is_test = True
        #if is_test:
        if 'test' in lidar_path:
            step_size = self.radius/4
            grid_size = 0.2
            num_points = 640000
            pts_semantic_gt = batch_data_samples[0].eval_ann_info['pts_semantic_mask']
            pts_instance_gt = batch_data_samples[0].eval_ann_info['pts_instance_mask']
            original_points = batch_inputs_dict['points'][0]
            regions = self.generate_cylindrical_regions(original_points, self.radius, step_size)
            num_cls = self.test_cfg.num_sem_cls        
            votes_counter = np.zeros(                    # (N_total, num_cls)
                (original_points.shape[0], num_cls),
                dtype=np.int16
            )
            all_pre_ins = np.full(original_points.shape[0], -1)
            max_instance = 0

            global_instance_scores = np.zeros((original_points.shape[0],), dtype=float) 

            best_masks = []

            all_instance_labels = set(np.unique(pts_instance_gt))
            
            ##########output_path = "work_dirs/bluepoint_th04fixed_03_priority_test_tobedelete"
            #########output_path = "work_dirs/bluepoint_forinstancev2"
            output_path = self.test_cfg.get('output_dir', 'work_dirs/default_output')
            score_th1 = self.score_th
            score_th2 = 0.3
            t2 = time.time()   
            last_results   = None    
            last_originids = None              
            #########print(f"generate regions: {(t2 - t1)*1000:.0f} ms")
            for region_idx, region in enumerate(tqdm(regions, desc="Processing regions")):
                t2 = time.time()                 
                region_mask = ((original_points[:, 0] - region[0]) ** 2 + (original_points[:, 1] - region[1]) ** 2) <= self.radius ** 2
                pc1 = original_points[region_mask]
                pc1_indices = torch.where(region_mask)[0]
                t3 = time.time()                 
                #########print(f"generate regions step2: {(t3 - t2)*1000:.0f} ms")
                if len(pc1) == 0:
                    continue

                pc2, pc2_indices = self.grid_sample(pc1, pc1_indices, grid_size)
                t4_1_1 = time.time()                 
                #########print(f"u net 1--1: {(t4_1_1 - t3)*1000:.0f} ms")
                if len(pc2) < num_points:
                    pc3 = pc2
                    pc3_indices = pc2_indices
                elif len(pc2) > num_points:
                    pc3, pc3_indices = self.points_random_sampling(pc2, pc2_indices, num_points)
                
                t4_1_2 = time.time()                 
                #########print(f"u net 1--2: {(t4_1_2 - t4_1_1)*1000:.0f} ms")

                coordinates, features, inverse_mapping2, spatial_shape = self.collate([pc3])
                t4_2 = time.time() 
                #########print(f"u net 2: {(t4_2 - t4_1_2)*1000:.0f} ms")
                x = spconv.SparseConvTensor(features, coordinates, spatial_shape, len(batch_data_samples))
                t4_3 = time.time() 
                #########print(f"u net 3: {(t4_3 - t4_2)*1000:.0f} ms")
                fine_feats, mid1_feats, mid2_feats = self.extract_feat(x)
                t4_4 = time.time() 
                #########print(f"u net 4: {(t4_4 - t4_3)*1000:.0f} ms")
                embed_logits = self.Embed(fine_feats[0])
                bi_semantic_logits = self.BiSemantic(fine_feats[0]) 
                
                wood_class = 1
                semantic_predictions_bi = torch.argmax(bi_semantic_logits, dim=1)
                tree_indices = torch.where(semantic_predictions_bi == wood_class)[0]  #all voxel
                t5 = time.time()                 
                #########print(f"u net heads: {(t5 - t4_4)*1000:.0f} ms")  
                with torch.no_grad():
                    nn_idx_pc1 = []                
                    chunk = self.chunk
                    for ss in range(0, pc1.shape[0], chunk):
                        ee = min(ss + chunk, pc1.shape[0])
                        nn_idx_pc1.append(
                            torch.cdist(pc1[ss:ee].float(), pc3.float()).argmin(1)
                        )
                    nn_idx_pc1 = torch.cat(nn_idx_pc1)   # (N_pc1,)  
                if tree_indices.numel() > 1:
                    
                    # FPS from all tree points
                    batch_tensor_4 = torch.zeros(embed_logits[tree_indices].size(0), dtype=torch.long).to(embed_logits.device)  # Ensure batch_tensor on same device
                    topk_indices_4 = fps(embed_logits[tree_indices], batch_tensor_4, ratio=min(self.query_point_num / embed_logits[tree_indices].size(0), torch.tensor([1.0]).to(embed_logits.device)))
                    selected_indices_case4 = tree_indices[topk_indices_4]

                    # add content queries
                    queries = []
                    queries.append(fine_feats[0][selected_indices_case4])
                    t6 = time.time()                 
                    #########print(f"generate queries: {(t6 - t5)*1000:.0f} ms")  
                    ms_feats = (fine_feats, mid1_feats, mid2_feats)
                    x_dec = self.decoder(ms_feats, queries)
                    t7 = time.time()                 
                    #########print(f"transformer decoder: {(t7 - t6)*1000:.0f} ms")  
                    results_list = self.predict_by_feat_test(x_dec, inverse_mapping2, pc3, selected_indices_case4)
                    t7_1 = time.time() 
                    #########print(f"postprocessing 1 step1: {(t7_1 - t7)*1000:.0f} ms")
                    # Collect masks and their scores, process them immediately
                    masks = results_list[0].pts_instance_mask[0]
                    scores = results_list[0].instance_scores
                    valid_scores_mask = scores > score_th1

                    # Prepare region for cylinder edge calculation
                    center = region  # assuming region is the cylinder's center (x, y)
                    edge_threshold = self.radius - 0.5  # Distance to cylinder edge threshold (0.5m from the edge)

                    # Compute distances to cylinder center for all points
                    pc3_distances = torch.sqrt((pc3[:, 0] - center[0]) ** 2 + (pc3[:, 1] - center[1]) ** 2)
                    t7_2 = time.time() 
                    #########print(f"postprocessing 1 step2: {(t7_2 - t7_1)*1000:.0f} ms")
                    valid_scores_mask2 = np.ones_like(valid_scores_mask, dtype=bool)  # Initialize mask (all True)
                    for i, mask in enumerate(masks):
                        if valid_scores_mask[i]:  # Only check valid masks (with scores > 0.6)
                            # Check if any point in the mask is too close to the edge
                            if torch.any(pc3_distances[mask] > edge_threshold):
                                valid_scores_mask2[i] = False  # Invalidate this mask if any point is near the edge
                    t8 = time.time()                 
                    #########print(f"postprocessing 1: {(t8 - t7)*1000:.0f} ms")  
                    ####### Apply both score and edge distance filtering
                    '''
                    final_valid_mask = valid_scores_mask & valid_scores_mask2

                    masks = masks[final_valid_mask]
                    scores = scores[final_valid_mask]

                    # Nearest neighbor mapping for masks to pc1
                    for mask, score in zip(masks, scores):
                        mask_pc1 = self.nearest_neighbor_mapping(pc1, pc3, mask)
                        mask_points = pc1_indices[mask_pc1].cpu().numpy()

                        # Vectorized update of global instance mask and scores
                        update_mask = score > global_instance_scores[mask_points]
                        global_instance_scores[mask_points[update_mask]] = score
                        all_pre_ins[mask_points[update_mask]] = max_instance

                        if np.any(update_mask):
                            # Add the new mask
                            best_masks.append((mask_points, max_instance, score))

                        #max_instance += 1'''

                    keep = torch.where(
                        torch.tensor(valid_scores_mask, device=pc3.device) &
                        torch.tensor(valid_scores_mask2, device=pc3.device)
                    )[0]                                            # (K,)

                    if keep.numel():
                        # mask/score（GPU）
                        masks_kept  = torch.as_tensor(masks,  device=pc3.device)[keep]   # (K,N3)
                        scores_kept = torch.as_tensor(scores, device=pc3.device)[keep]   # (K,)

                        # ② voxel → pc1   (K, N_pc1)  → COO
                        rows_list = []
                        cols_list = []
                        max_chunk = (torch.iinfo(torch.int32).max // masks_kept.shape[0]) - 1_000_000
                        max_chunk = max(1, max_chunk)
                        chunk_size = min(nn_idx_pc1.shape[0], max_chunk)

                        for start in range(0, nn_idx_pc1.shape[0], chunk_size):
                            end = min(start + chunk_size, nn_idx_pc1.shape[0])
                            mk_bool_chunk = masks_kept[:, nn_idx_pc1[start:end]]
                            rows_chunk, cols_chunk = mk_bool_chunk.nonzero(as_tuple=True)
                            cols_chunk = cols_chunk + start
                            rows_list.append(rows_chunk)
                            cols_list.append(cols_chunk)

                        rows = torch.cat(rows_list, dim=0)
                        cols = torch.cat(cols_list, dim=0)     
                        score_per_hit = scores_kept[rows]               # (nnz,)

                        N1 = pc1.shape[0]
                        best_score = torch.full((N1,), -1., device=pc3.device)
                        best_mid   = torch.full((N1,), -1 , dtype=torch.long, device=pc3.device)

                        best_score.index_reduce_(0, cols, score_per_hit, reduce='amax')
                        improved_mask = score_per_hit == best_score[cols]
                        best_mid.index_put_((cols[improved_mask],),
                                            rows[improved_mask], accumulate=False)

                        pts_glob   = pc1_indices.cpu().numpy()
                        new_scores = best_score.cpu().numpy()
                        better_pts = new_scores > global_instance_scores[pts_glob]

                        if better_pts.any():
                            global_instance_scores[pts_glob[better_pts]] = new_scores[better_pts]
                            all_pre_ins[pts_glob[better_pts]] = (
                                max_instance + best_mid.cpu().numpy()[better_pts]
                            )

                            mids_unique = np.unique(best_mid.cpu().numpy()[better_pts])
                            for mid in mids_unique:
                                sel_pts = (cols[rows == mid]).cpu().numpy()      
                                best_masks.append(
                                    (pts_glob[sel_pts],
                                    max_instance + int(mid),
                                    float(scores_kept[int(mid)]))
                                )

                        max_instance += int(masks_kept.size(0))        
                    t9 = time.time()                 
                    #########print(f"postprocessing 2: {(t9 - t8)*1000:.0f} ms")  
                    #cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, results_list[0].pts_semantic_mask[0])
                    sem_pred_pc3 = results_list[0].pts_semantic_mask[0]
                    if isinstance(sem_pred_pc3, np.ndarray):
                        sem_pred_pc3 = torch.from_numpy(sem_pred_pc3).to(pc3.device)
                    else:
                        sem_pred_pc3 = sem_pred_pc3.to(pc3.device)
                    cylinder_current_semantic_pre = sem_pred_pc3[nn_idx_pc1]

                    #originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                    #all_pre_sem = self.vote_semantic_labels(all_pre_sem, torch.where(region_mask)[0], cylinder_current_semantic_pre)
                    ids_np = torch.where(region_mask)[0].cpu().numpy()                 # (N_pc1,)
                    sem_np = cylinder_current_semantic_pre.cpu().numpy().astype(int)   # (N_pc1,)

                    np.add.at(votes_counter, (ids_np, sem_np), 1) 

                    originids = pc3_indices.cpu().numpy()  # Use pc3_indices for ground truth labels
                    last_results = results_list      
                    last_originids = originids  
                    last_batch_len = len(batch_data_samples)      
                    t10 = time.time()                 
                    #########print(f"postprocessing 3: {(t10 - t9)*1000:.0f} ms") 
                    
                    '''
                    originids = pc3_indices.cpu().numpy()  # Use pc3_indices for ground truth labels
                    # Get gt labels for pc3
                    pc3_semantic_gt = pts_semantic_gt[originids]
                    pc3_instance_gt = pts_instance_gt[originids]
                    # Save each pc3 to a separate .ply file
                    region_dir = f"work_dirs/to_be_delete/{current_filename}/region_0"
                    region_dir = os.path.join(output_path, current_filename, f"region_0")
                    region_ply_path = os.path.join(region_dir, "pc_ins_sem.ply")
                    self.save_ply(pc3.cpu().numpy(), results_list[0].pts_semantic_mask[0], results_list[0].pts_instance_mask[1], region_ply_path, pc3_semantic_gt, pc3_instance_gt)
                    '''
                else:
                    projected_semantic_logits = bi_semantic_logits[inverse_mapping2]
                    semantic_predictions_pc3 = torch.argmax(projected_semantic_logits, dim=1)
                    #cylinder_current_semantic_pre = self.nearest_neighbor_mapping(pc1, pc3, semantic_predictions_pc3)
                    proj_logits   = bi_semantic_logits[inverse_mapping2]         # (N_pc3, C)
                    sem_pred_pc3  = torch.argmax(proj_logits, dim=1) 
                    if isinstance(sem_pred_pc3, np.ndarray):
                        sem_pred_pc3 = torch.from_numpy(sem_pred_pc3).to(pc3.device)
                    else:
                        sem_pred_pc3 = sem_pred_pc3.to(pc3.device)
                    cylinder_current_semantic_pre = sem_pred_pc3[nn_idx_pc1]   
                    #originids = torch.where(region_mask)[0].cpu().numpy()  # Move to CPU before using np.where
                    #all_pre_sem = self.vote_semantic_labels(all_pre_sem, torch.where(region_mask)[0], cylinder_current_semantic_pre)                    
                    ids_np = torch.where(region_mask)[0].cpu().numpy()                 # (N_pc1,)
                    sem_np = cylinder_current_semantic_pre.cpu().numpy().astype(int)   # (N_pc1,)

                    np.add.at(votes_counter, (ids_np, sem_np), 1) 
                    t10 = time.time()                 
                    #########print(f"postprocessing 4: {(t10 - t5)*1000:.0f} ms") 
                del pc1, pc1_indices, pc2, pc2_indices, pc3, pc3_indices
                del embed_logits, bi_semantic_logits
                torch.cuda.empty_cache() 
                import gc
                gc.collect()
     
            # Post-processing step
            #final_semantic_labels = self.finalize_semantic_labels_old(all_pre_sem)
            #ground_mask = (final_semantic_labels == 0)

            final_semantic_labels = votes_counter.argmax(1)         # (N_total,)
            final_semantic_labels[votes_counter.sum(1) == 0] = -1  
            ground_mask = (final_semantic_labels == 0)

            all_pre_ins[ground_mask] = -1
            t11 = time.time()                 
            #print(f"postprocessing 5: {(t11 - t10)*1000:.0f} ms") 
            # Remove instances with fewer than 10 points
            #unique_instances, instance_counts = np.unique(all_pre_ins, return_counts=True)
            #small_instances = unique_instances[instance_counts < 10]
            #for instance in small_instances:
            #    all_pre_ins[all_pre_ins == instance] = -1
            t11 = time.time()     
            # Remove instances with fewer than 10 points  
            uniq, cnt = np.unique(all_pre_ins, return_counts=True)
            to_kill   = np.isin(all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])
            all_pre_ins[to_kill] = -1
            t12 = time.time()
            print(f"postprocessing 6: {(t12 - t11)*1000:.0f} ms")
            # Remove replaced old masks
            unique_best_masks = []
            for mask_points, instance_id, score in best_masks:
                if np.any(all_pre_ins[mask_points] == instance_id):
                    unique_best_masks.append((mask_points, instance_id, score))
            t13 = time.time()                 
            print(f"postprocessing 7: {(t13 - t12)*1000:.0f} ms")

            clean_all_pre_ins, merged_masks, merged_instance_scores = self.merge_overlapping_instances_by_score_speedup(all_pre_ins, unique_best_masks,overlap_threshold=score_th2)
            t14 = time.time()                 
            print(f"postprocessing 8: {(t14 - t13)*1000:.0f} ms")
            # Re-label instances to ensure continuous labeling
            unique_labels = np.unique(clean_all_pre_ins)
            unique_labels = unique_labels[unique_labels >= 0]  # Exclude background label (-1)
            relabel_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            relabel_map[-1] = -1  # Keep background as -1
            clean_all_pre_ins = np.vectorize(relabel_map.get)(clean_all_pre_ins)
            t14 = time.time()                 
            ######print(f"postprocessing 8: {(t14 - t13)*1000:.0f} ms")
            # Save the final combined results
            region_path = os.path.join(output_path, f"{current_filename}.ply")
 
            self.save_ply_withscore(original_points.cpu().numpy(), final_semantic_labels, clean_all_pre_ins, merged_instance_scores, region_path, pts_semantic_gt, pts_instance_gt)
            #self.save_bluepoints(original_points.cpu().numpy(), final_semantic_labels, clean_all_pre_ins, merged_instance_scores, region_path, pts_semantic_gt, pts_instance_gt)
            
            t15 = time.time()                 
            ######print(f"postprocessing 9: {(t15 - t14)*1000:.0f} ms")
            #for i, data_sample in enumerate(batch_data_samples):
            #    data_sample.pred_pts_seg = results_list[i]
            #    data_sample.pred_pts_seg['originids'] = originids
                #data_sample.originids = originids
            if last_results is not None and len(last_results)==len(batch_data_samples):                 # 本帧里至少有一个 region 得到了结果
                for i, data_sample in enumerate(batch_data_samples):
                    data_sample.pred_pts_seg = last_results[i]
                    data_sample.pred_pts_seg['originids'] = last_originids
            else:                                      
                for data_sample in batch_data_samples:
                    data_sample.pred_pts_seg = None
            return batch_data_samples
        else:
            coordinates, features, inverse_mapping, spatial_shape = self.collate(
                batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape, len(batch_data_samples))

            fine_feats, mid1_feats, mid2_feats = self.extract_feat(x)

            queries = []
            for i in range(len(fine_feats)):
                max_len = min(self.query_point_num, len(fine_feats[i]))
                queries.append(fine_feats[i][0:max_len])
            
            ms_feats = (fine_feats, mid1_feats, mid2_feats)
            x_dec = self.decoder(ms_feats, queries)

            results_list = self.predict_by_feat(x_dec, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]

            return batch_data_samples