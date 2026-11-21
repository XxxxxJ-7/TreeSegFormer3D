"""Multi-Scale Cross-Attention V2: Per-Scale Attention Masks.

Problem in V1
-------------
The original MultiScaleCrossAttentionLayer only applies the attention mask
to the finest scale (scale 0).  Coarser scales (scale 1 at stride-2,
scale 2 at stride-4) receive **no mask at all**, so every query attends
indiscriminately to ALL coarse-scale points.  This causes multiple
queries to receive nearly identical information from the coarse scales,
producing redundant / conflicting instance masks and a catastrophic
Precision drop (90 → 67).

V2 Fix
------
For each decoder layer, after the fine-scale attention mask is generated
by ``_forward_head`` (from the predicted masks), it is **downsampled**
to every coarser scale using the known stride relationship between
voxel coordinates:

    fine voxel (z, y, x)  →  coarse voxel (z//s, y//s, x//s)

Aggregation rule (conservative):
    A coarse voxel is **masked** only if ALL of its fine-scale children
    are masked.  If ANY child is unmasked, the coarse voxel is unmasked.

This ensures each query focuses on its predicted region at ALL scales,
eliminating the redundant cross-scale information that caused false
positive instances.

New components:
    downsample_attn_mask                              helper
    MultiScaleCrossAttentionLayer_V2                  per-scale masks
    ForAINetv2QueryDecoder_XAwarequery_MultiScale_V2  mask generation
    ForAINetV2OneFormer3D_XAwarequery_MultiScale_V2   coord extraction
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np

from torch_scatter import scatter_add
from torch_cluster import fps

from mmdet3d.registry import MODELS

from .multi_scale_cross_attn import (
    MultiScaleCrossAttentionLayer,
    ForAINetv2QueryDecoder_XAwarequery_MultiScale,
    ForAINetV2OneFormer3D_XAwarequery_MultiScale,
)
from .panoptic_losses import discriminative_loss


# =========================================================================
#  Attention-mask downsampling  (fine → coarse)
# =========================================================================

def downsample_attn_mask(attn_mask, fine_coords, coarse_coords, stride):
    """Downsample a fine-scale boolean attention mask to a coarser scale.

    Uses the known stride relationship: a fine voxel at (z, y, x) maps
    to the coarse voxel at (z // s, y // s, x // s).

    Aggregation rule:
        coarse masked  ←→  ALL fine children masked
        coarse unmasked ←→  ANY fine child unmasked

    Args:
        attn_mask (Tensor): Fine-scale mask, ``(Q, N_fine)``, bool.
            True = "do NOT attend".
        fine_coords (Tensor): Fine voxel integer coords, ``(N_fine, 3)``.
        coarse_coords (Tensor): Coarse voxel integer coords, ``(N_coarse, 3)``.
        stride (int): Spatial stride between fine and coarse.

    Returns:
        Tensor: Coarse-scale mask, ``(Q, N_coarse)``, bool.
    """
    device = attn_mask.device
    n_queries, n_fine = attn_mask.shape
    n_coarse = coarse_coords.size(0)

    if n_fine == 0 or n_coarse == 0:
        return torch.zeros(n_queries, n_coarse,
                           dtype=torch.bool, device=device)

    # ---- map fine coords into the coarse coordinate system ----
    fine_expected = fine_coords.long() // stride          # (N_fine, 3)

    max_coord = torch.max(
        coarse_coords.max(0).values,
        fine_expected.max(0).values
    ) + 1

    mult = torch.tensor([
        max_coord[1] * max_coord[2],
        max_coord[2],
        1
    ], device=device, dtype=torch.long)

    coarse_keys = (coarse_coords.long() * mult).sum(1)   # (N_coarse,)
    fine_keys = (fine_expected * mult).sum(1)              # (N_fine,)

    # ---- build lookup:  encoded-coord  →  coarse index ----
    max_key = int(max(coarse_keys.max().item(),
                      fine_keys.max().item())) + 1
    key_to_coarse = torch.full((max_key,), -1,
                               dtype=torch.long, device=device)
    key_to_coarse[coarse_keys] = torch.arange(n_coarse, device=device)

    fine_parent = key_to_coarse[fine_keys.clamp(max=max_key - 1)]
    valid = fine_parent >= 0                              # (N_fine,)

    if not valid.any():
        return torch.zeros(n_queries, n_coarse,
                           dtype=torch.bool, device=device)

    valid_fine_idx = torch.where(valid)[0]
    valid_parents = fine_parent[valid_fine_idx]            # coarse idx

    # ---- aggregate: count unmasked fine children per coarse voxel ----
    unmasked_fine = (~attn_mask[:, valid_fine_idx]).float()  # (Q, n_valid)

    unmasked_count = torch.zeros(n_queries, n_coarse, device=device)
    unmasked_count.scatter_add_(
        1,
        valid_parents.unsqueeze(0).expand(n_queries, -1),
        unmasked_fine,
    )

    # coarse masked iff ALL children masked  (unmasked_count == 0)
    coarse_mask = (unmasked_count == 0)

    # safety: if a query has everything masked → unmask all
    all_masked = coarse_mask.all(dim=-1)
    coarse_mask[all_masked] = False

    return coarse_mask


# =========================================================================
#  V2 Cross-Attention Layer  (per-scale masks)
# =========================================================================

class MultiScaleCrossAttentionLayer_V2(MultiScaleCrossAttentionLayer):
    """Multi-scale cross-attention with **per-scale** attention masks.

    Compared to V1 (mask only on scale 0), V2 receives a list of masks,
    one per scale, so every scale is properly guided.
    """

    def forward(self, multi_sources, queries, multi_attn_masks=None):
        """
        Args:
            multi_sources: ``[num_scales][batch][n_pts_s, d_model]``.
            queries:       ``[batch][n_queries, d_model]``.
            multi_attn_masks: ``[num_scales]`` list.  Each element is
                either ``None`` or ``[batch][n_queries, n_pts_s]`` bool.

        Returns:
            ``[batch][n_queries, d_model]``.
        """
        outputs = []
        for i in range(len(queries)):
            scale_outputs = []
            for s in range(self.num_scales):
                k = v = multi_sources[s][i]
                # V2: each scale gets its own mask
                if (multi_attn_masks is not None
                        and multi_attn_masks[s] is not None):
                    mask = multi_attn_masks[s][i]
                else:
                    mask = None
                out, _ = self.attns[s](queries[i], k, v, attn_mask=mask)
                scale_outputs.append(out)

            if self.fusion_mode == 'concat':
                fused = torch.cat(scale_outputs, dim=-1)
                fused = self.fusion(fused)
            elif self.fusion_mode == 'weighted_sum':
                w = torch.softmax(self.scale_weights, dim=0)
                fused = sum(wgt * so for wgt, so in zip(w, scale_outputs))

            if self.fix:
                fused = self.dropout(fused)
            fused = fused + queries[i]
            if self.fix:
                fused = self.norm(fused)
            outputs.append(fused)

        return outputs


# =========================================================================
#  V2 Decoder
# =========================================================================

@MODELS.register_module()
class ForAINetv2QueryDecoder_XAwarequery_MultiScale_V2(
        ForAINetv2QueryDecoder_XAwarequery_MultiScale):
    """Multi-scale decoder V2 with attention masks for **all** scales.

    The model's ``extract_feat`` stores voxel coordinates on this decoder
    as ``self._scale_coords = (fine_coords_list, mid1_coords_list, …)``.

    During iterative prediction the fine-scale attention mask is
    downsampled to every coarser scale before being passed to
    ``MultiScaleCrossAttentionLayer_V2``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace V1 cross-attention layers with V2 (per-scale masks)
        num_layers = len(self.cross_attn_layers)
        self.cross_attn_layers = nn.ModuleList([
            MultiScaleCrossAttentionLayer_V2(
                kwargs['d_model'], kwargs['num_heads'], kwargs['dropout'],
                num_scales=kwargs.get('num_scales', 3),
                fusion_mode=kwargs.get('fusion_mode', 'concat'),
                fix=kwargs.get('fix_attention', True))
            for _ in range(num_layers)
        ])

    # ---- mask generation ------------------------------------------------

    def _build_multi_scale_masks(self, fine_attn_masks):
        """Generate per-scale masks from the fine-scale masks.

        Args:
            fine_attn_masks: ``List[Tensor]``, length = batch_size.
                Each tensor ``(Q, N_fine_i)`` bool.

        Returns:
            List of length ``num_scales``.  Element *s* is either
            ``None`` or ``List[Tensor]`` of length batch_size with
            coarse masks at stride 2^s.
        """
        scale_coords = getattr(self, '_scale_coords', None)
        if scale_coords is None or fine_attn_masks is None:
            return None

        fine_coords_list = scale_coords[0]
        batch_size = len(fine_attn_masks)

        if len(fine_coords_list) != batch_size:
            return None

        # scale 0 → fine masks unchanged
        multi_masks = [fine_attn_masks]

        # scale 1, 2, … → downsample from fine
        strides = [2, 4, 8, 16]
        for s_idx in range(1, self.num_scales):
            if s_idx >= len(scale_coords):
                multi_masks.append(None)
                continue
            coarse_coords_list = scale_coords[s_idx]
            stride = strides[s_idx - 1]
            coarse_masks = []
            for i in range(batch_size):
                coarse_masks.append(downsample_attn_mask(
                    fine_attn_masks[i],
                    fine_coords_list[i],
                    coarse_coords_list[i],
                    stride=stride))
            multi_masks.append(coarse_masks)

        return multi_masks

    # ---- iterative prediction (main path) --------------------------------

    def forward_iter_pred(self, x, queries):
        """Iterative prediction with multi-scale attention masks."""
        pred_scores, pred_masks = [], []

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
            # V2: build masks for ALL scales, not just scale 0
            multi_attn_masks = self._build_multi_scale_masks(attn_mask)

            queries = self.cross_attn_layers[i](
                multi_inst_feats, queries, multi_attn_masks)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)

            pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'masks': m, 'scores': s}
            for s, m in zip(pred_scores[:-1], pred_masks[:-1])]
        return dict(
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


# =========================================================================
#  V2 Model
# =========================================================================

@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_MultiScale_V2(
        ForAINetV2OneFormer3D_XAwarequery_MultiScale):
    """Multi-scale model V2 with attention masks for all scales.

    Differences from V1:

    1. ``extract_feat`` also extracts voxel **coordinates** for each
       scale and stores them on the decoder (``decoder._scale_coords``).
       The decoder uses them to downsample the fine-scale attention mask
       to coarser scales.

    2. ``loss`` synchronises the stored coordinates when batch elements
       are filtered (only relevant for batch_size > 1).
    """

    def extract_feat(self, x):
        """Extract multi-scale features **and** voxel coordinates."""
        x = self.input_conv(x)
        output, blocks = self.unet(x)
        x_out = self.output_layer(output)

        fine_list, fine_coords = [], []
        for i in x_out.indices[:, 0].unique():
            mask = x_out.indices[:, 0] == i
            fine_list.append(x_out.features[mask])
            fine_coords.append(x_out.indices[mask, 1:])

        mid1_tensor = blocks[-2]
        mid1_list, mid1_coords = [], []
        for i in mid1_tensor.indices[:, 0].unique():
            mask = mid1_tensor.indices[:, 0] == i
            mid1_list.append(mid1_tensor.features[mask])
            mid1_coords.append(mid1_tensor.indices[mask, 1:])

        mid2_tensor = blocks[-3]
        mid2_list, mid2_coords = [], []
        for i in mid2_tensor.indices[:, 0].unique():
            mask = mid2_tensor.indices[:, 0] == i
            mid2_list.append(mid2_tensor.features[mask])
            mid2_coords.append(mid2_tensor.indices[mask, 1:])

        # Store on decoder for mask downsampling in forward_iter_pred
        self.decoder._scale_coords = (
            fine_coords, mid1_coords, mid2_coords)

        return fine_list, mid1_list, mid2_list

    # ---- coord sync helper (batch filtering in loss) --------------------

    def _sync_decoder_coords(self, original_indices):
        """Filter decoder's stored coords to match a filtered batch."""
        full = self.decoder._scale_coords
        self.decoder._scale_coords = tuple(
            [full[s][j] for j in original_indices]
            for s in range(len(full))
        )

    # ---- loss (copied from V1, one line added for coord sync) -----------

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses.

        Identical to the V1 parent except that when batch elements are
        filtered (batch_size > 1 with empty queries), the decoder's
        stored scale coordinates are filtered in sync.
        """
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        fine_feats, mid1_feats, mid2_feats = self.extract_feat(x)

        embed_logits = [self.Embed(y) for y in fine_feats]
        bi_semantic_logits = [self.BiSemantic(y) for y in fine_feats]

        total_discriminative_loss = 0
        total_semantic_loss_bi = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][
                inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]

            pts_instance_mask = \
                batch_data_samples[i].gt_pts_seg.pts_instance_mask
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

        queries = []
        queries_inslabel = []
        queries_idx = []

        if self.prepare_epoch:
            cur_epoch = kwargs.get('epoch', self._train_epoch)
            if cur_epoch > self.prepare_epoch:
                for i in range(batch_size):
                    voxel_superpoints = inverse_mapping[coordinates[:, 0][
                        inverse_mapping] == i]
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

                        batch_tensor_4 = torch.zeros(
                            embed_logits[i][tree_indices].size(0),
                            dtype=torch.long).to(embed_logits[i].device)
                        topk_indices_4 = fps(
                            embed_logits[i][tree_indices], batch_tensor_4,
                            ratio=min(
                                self.query_point_num /
                                embed_logits[i][tree_indices].size(0),
                                torch.tensor([1.0]).to(
                                    embed_logits[i].device)))
                        selected_indices_case4 = tree_indices[topk_indices_4]

                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(
                            current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()),
                            voxel_superpoints, dim=0)
                        avg_points = current_points_add / \
                            voxel_counts.unsqueeze(-1).clamp(min=1)

                        queries.append(
                            fine_feats[i][selected_indices_case4])

                        pts_instance_mask = batch_data_samples[
                            i].gt_pts_seg.pts_instance_mask
                        voxel_instance_labels = \
                            self.get_voxel_instance_labels(
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
                             batch_data_samples[i],
                             queries_inslabel[i], i)
                            for i in range(len(queries))
                            if len(queries[i]) > 0]
                        (fine_feats_f, queries_f,
                         batch_data_samples_f,
                         queries_inslabel_f,
                         original_indices) = zip(*filtered_results)
                        fine_feats_f = list(fine_feats_f)
                        queries_f = list(queries_f)
                        batch_data_samples_f = list(batch_data_samples_f)
                        queries_inslabel_f = list(queries_inslabel_f)
                        original_indices = list(original_indices)

                        mid1_feats_f = [mid1_feats[j]
                                        for j in original_indices]
                        mid2_feats_f = [mid2_feats[j]
                                        for j in original_indices]

                        # ---- V2: sync decoder coords after filtering ----
                        self._sync_decoder_coords(original_indices)
                    else:
                        fine_feats_f = fine_feats
                        queries_f = queries
                        batch_data_samples_f = batch_data_samples
                        queries_inslabel_f = queries_inslabel
                        original_indices = list(
                            range(len(batch_data_samples)))
                        mid1_feats_f = mid1_feats
                        mid2_feats_f = mid2_feats

                    ms_feats = (fine_feats_f, mid1_feats_f, mid2_feats_f)
                    x_dec = self.decoder(ms_feats, queries_f)

                    sp_gt_instances = []
                    for i in range(len(batch_data_samples_f)):
                        voxel_superpoints = inverse_mapping[
                            coordinates[:, 0][inverse_mapping]
                            == original_indices[i]]
                        voxel_superpoints = torch.unique(
                            voxel_superpoints, return_inverse=True)[1]
                        inst_mask = batch_data_samples_f[
                            i].gt_pts_seg.pts_instance_mask
                        sem_mask = batch_data_samples_f[
                            i].gt_pts_seg.pts_semantic_mask
                        assert voxel_superpoints.shape == inst_mask.shape

                        batch_data_samples_f[
                            i].gt_instances_3d.sp_sem_masks = \
                            self.get_gt_semantic_masks(
                                sem_mask, voxel_superpoints,
                                self.num_classes)
                        batch_data_samples_f[
                            i].gt_instances_3d.sp_inst_masks = \
                            self.get_gt_inst_masks(
                                inst_mask, voxel_superpoints)

                        (batch_data_samples_f[
                             i].gt_instances_3d.labels_3d,
                         batch_data_samples_f[
                             i].gt_instances_3d.sp_inst_masks,
                         batch_data_samples_f[
                             i].gt_instances_3d.ratio_inspoint
                         ) = self.filter_stuff_masks(
                            batch_data_samples_f[i].gt_instances_3d,
                            self.stuff_classes,
                            batch_data_samples_f[
                                i].gt_pts_seg.ratio_inspoint)

                        batch_data_samples_f[
                            i].gt_instances_3d.query_inslabel = \
                            queries_inslabel_f[i]
                        sp_gt_instances.append(
                            batch_data_samples_f[i].gt_instances_3d)

                    loss = self.criterion(x_dec, sp_gt_instances)
                    loss_final.update(loss)
        return loss_final
