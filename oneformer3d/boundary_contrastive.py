"""Boundary Contrastive Learning for Instance Segmentation.

Enhances embedding discrimination at instance boundaries by applying
stronger contrastive supervision to spatially adjacent voxels that
belong to different instances. This forces the model to learn sharper
embedding transitions at tree boundaries, which is critical in dense
forests where canopies overlap.

Key idea:
  - Find voxel pairs that are spatial neighbors but belong to different
    instances (cross-boundary pairs)
  - Push their embeddings apart (repulsive loss)
  - Find voxel pairs that are spatial neighbors and belong to the same
    instance (intra-instance boundary pairs)
  - Pull their embeddings together (attractive loss)
  - Use hard-negative mining to focus on the most confusing pairs

Only the loss computation changes; backbone, decoder, and predict
are all inherited unchanged from the IAS variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import numpy as np
from torch_cluster import radius_graph

from torch_scatter import scatter_add, scatter_mean
from mmdet3d.registry import MODELS

from .instance_aware_sampling import ForAINetV2OneFormer3D_XAwarequery_IAS
from .panoptic_losses import discriminative_loss


def compute_boundary_contrastive_loss(
    embed_logits,
    voxel_coords,
    instance_labels,
    spatial_neighbor_radius=2.0,
    margin=1.5,
    temperature=0.1,
    max_pairs=4096,
    hard_negative_ratio=0.5,
    max_neighbors=32,
):
    """Compute boundary contrastive loss on voxel embeddings.

    Args:
        embed_logits (Tensor): Embedding features for foreground voxels,
            shape ``(N, D)``.
        voxel_coords (Tensor): 3D spatial coordinates of foreground voxels,
            shape ``(N, 3)``. These should be in real-world scale (not
            voxelized integers) for meaningful distance computation.
        instance_labels (Tensor): Instance label for each voxel,
            shape ``(N,)``.
        spatial_neighbor_radius (float): Maximum spatial distance to
            consider two voxels as neighbors.
        margin (float): Margin for the repulsive (inter-instance) loss.
        temperature (float): Temperature for the InfoNCE-style attractive
            loss component.
        max_pairs (int): Maximum number of boundary pairs to sample
            (for memory efficiency).
        hard_negative_ratio (float): Fraction of pairs selected via
            hard-negative mining (rest are random).
        max_neighbors (int): Maximum spatial neighbors retained per voxel.
            This bounds neighbor-search memory for large training crops.

    Returns:
        dict: Dictionary with 'bcl_repulsive' and 'bcl_attractive' losses.
    """
    device = embed_logits.device
    N = embed_logits.size(0)

    if N < 2:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return {'bcl_repulsive': zero, 'bcl_attractive': zero}

    # --- Step 1: Find spatial neighbor pairs efficiently ---
    # The previous chunked cdist still materialized an (8192 x N) matrix;
    # at 12 m crop radius this can require several GiB. radius_graph keeps
    # only a bounded local neighborhood, making memory O(N * neighbors).
    edge_index = radius_graph(
        voxel_coords.detach().float(),
        r=float(spatial_neighbor_radius),
        batch=None,
        loop=False,
        max_num_neighbors=int(max_neighbors),
    )
    if edge_index.numel() == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return {'bcl_repulsive': zero, 'bcl_attractive': zero}

    row_idx, col_idx = edge_index[0], edge_index[1]

    # Remove duplicates (i,j) and (j,i) by keeping only i < j
    valid = row_idx < col_idx
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]

    if row_idx.numel() == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return {'bcl_repulsive': zero, 'bcl_attractive': zero}

    # --- Step 2: Classify pairs as cross-boundary or intra-instance ---
    labels_i = instance_labels[row_idx]
    labels_j = instance_labels[col_idx]
    cross_boundary = labels_i != labels_j  # different instances
    intra_instance = labels_i == labels_j  # same instance

    # --- Step 3: Sample pairs (with hard-negative mining) ---
    cross_indices = torch.where(cross_boundary)[0]
    intra_indices = torch.where(intra_instance)[0]

    # Sample cross-boundary pairs (hard negatives: closest in embedding)
    # Bound the hard-mining candidate pool before gathering embeddings.
    # Sampling from an 8x pool preserves useful hard-negative diversity
    # without allocating tensors for every edge in a large radius graph.
    hard_pool_size = max(int(max_pairs) * 8, int(max_pairs))
    if cross_indices.numel() > hard_pool_size:
        pool_perm = torch.randperm(
            cross_indices.numel(), device=device)[:hard_pool_size]
        cross_indices = cross_indices[pool_perm]

    n_cross = min(max_pairs, cross_indices.numel())
    if n_cross > 0 and cross_indices.numel() > n_cross:
        n_hard = int(n_cross * hard_negative_ratio)
        n_random = n_cross - n_hard

        # Hard negatives: pairs with smallest embedding distance
        with torch.no_grad():
            cross_embed_dist = F.pairwise_distance(
                embed_logits[row_idx[cross_indices]],
                embed_logits[col_idx[cross_indices]]
            )
            _, hard_sort_idx = cross_embed_dist.sort()
            hard_sel = hard_sort_idx[:n_hard]

            # Random selection for the rest
            remaining = hard_sort_idx[n_hard:]
            if remaining.numel() > 0 and n_random > 0:
                rand_perm = torch.randperm(
                    remaining.numel(), device=device)[:n_random]
                rand_sel = remaining[rand_perm]
                cross_sel = torch.cat([hard_sel, rand_sel])
            else:
                cross_sel = hard_sel
        cross_indices = cross_indices[cross_sel]
    elif n_cross > 0:
        pass  # use all

    # Sample intra-instance pairs (random)
    n_intra = min(max_pairs, intra_indices.numel())
    if n_intra > 0 and intra_indices.numel() > n_intra:
        perm = torch.randperm(intra_indices.numel(), device=device)[:n_intra]
        intra_indices = intra_indices[perm]

    # --- Step 4: Compute repulsive loss (cross-boundary pairs) ---
    if cross_indices.numel() > 0:
        embed_i = embed_logits[row_idx[cross_indices]]
        embed_j = embed_logits[col_idx[cross_indices]]
        embed_dist = F.pairwise_distance(embed_i, embed_j, p=2)
        # Hinge loss: push apart until margin
        repulsive_loss = F.relu(margin - embed_dist).pow(2).mean()
    else:
        repulsive_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # --- Step 5: Compute attractive loss (intra-instance boundary pairs) ---
    if intra_indices.numel() > 0:
        embed_i = embed_logits[row_idx[intra_indices]]
        embed_j = embed_logits[col_idx[intra_indices]]
        embed_dist = F.pairwise_distance(embed_i, embed_j, p=2)
        # Pull together: penalize large distances within same instance
        attractive_loss = embed_dist.pow(2).mean()
    else:
        attractive_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return {
        'bcl_repulsive': repulsive_loss,
        'bcl_attractive': attractive_loss,
    }


# =========================================================================
#  Model with Instance-Aware Sampling + Boundary Contrastive Learning
# =========================================================================

@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_IAS_BCL(
        ForAINetV2OneFormer3D_XAwarequery_IAS):
    """IAS model augmented with Boundary Contrastive Learning.

    Adds a boundary contrastive loss to the first-stage training
    (embedding learning phase), which strengthens the embedding
    discrimination specifically at instance boundaries.

    Extra config keys:
        bcl_weight (float): Weight for the boundary contrastive loss.
            Default 0.5.
        bcl_margin (float): Margin for repulsive loss. Default 1.5.
        bcl_neighbor_radius (float): Spatial radius (in voxel units)
            to define neighbors. Default 2.0.
        bcl_max_pairs (int): Max boundary pairs per sample. Default 4096.
        bcl_max_neighbors (int): Max spatial neighbors per voxel. Default 32.
        bcl_temperature (float): Temperature for attractive loss.
            Default 0.1.
        bcl_hard_neg_ratio (float): Ratio of hard negatives. Default 0.5.
    """

    def __init__(
        self,
        bcl_weight=0.5,
        bcl_margin=1.5,
        bcl_neighbor_radius=2.0,
        bcl_max_pairs=4096,
        bcl_max_neighbors=32,
        bcl_temperature=0.1,
        bcl_hard_neg_ratio=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bcl_weight = bcl_weight
        self.bcl_margin = bcl_margin
        self.bcl_neighbor_radius = bcl_neighbor_radius
        self.bcl_max_pairs = bcl_max_pairs
        self.bcl_max_neighbors = bcl_max_neighbors
        self.bcl_temperature = bcl_temperature
        self.bcl_hard_neg_ratio = bcl_hard_neg_ratio

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Loss with additional boundary contrastive learning.

        The boundary contrastive loss is applied to the embedding
        features during the first training stage (alongside the
        discriminative loss), enhancing boundary discrimination.
        """
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
        total_bcl_repulsive = 0
        total_bcl_attractive = 0
        batch_size = len(batch_data_samples)

        for i in range(batch_size):
            voxel_superpoints = inverse_mapping[
                coordinates[:, 0][inverse_mapping] == i]
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

            # --- Original discriminative loss ---
            batch_v = torch.full_like(voxel_instance_labels, i)
            discriminative_losses = discriminative_loss(
                filtered_embed_logits, voxel_instance_labels, batch_v, 5)

            for loss_name, loss in discriminative_losses.items():
                total_discriminative_loss += (
                    loss if loss_name == "ins_loss" else 0)

            # --- Boundary Contrastive Loss ---
            # Compute voxel 3D coordinates (average point position per voxel)
            current_points = batch_inputs_dict['points'][i][:, :3]
            voxel_coords_sum = scatter_add(
                current_points[valid_instance_mask],
                voxel_superpoints[valid_instance_mask], dim=0)
            voxel_counts = scatter_add(
                torch.ones(valid_instance_mask.sum().item(),
                           device=device).float()
                if isinstance(valid_instance_mask.sum(), torch.Tensor)
                else torch.ones(int(valid_instance_mask.sum()),
                                device=device).float(),
                voxel_superpoints[valid_instance_mask], dim=0)
            # Only take valid voxels
            voxel_coords_avg = (
                voxel_coords_sum[valid_voxel_indices] /
                voxel_counts[valid_voxel_indices].unsqueeze(-1).clamp(min=1))

            bcl_losses = compute_boundary_contrastive_loss(
                embed_logits=filtered_embed_logits,
                voxel_coords=voxel_coords_avg,
                instance_labels=voxel_instance_labels,
                spatial_neighbor_radius=self.bcl_neighbor_radius,
                margin=self.bcl_margin,
                temperature=self.bcl_temperature,
                max_pairs=self.bcl_max_pairs,
                hard_negative_ratio=self.bcl_hard_neg_ratio,
                max_neighbors=self.bcl_max_neighbors,
            )
            total_bcl_repulsive += bcl_losses['bcl_repulsive']
            total_bcl_attractive += bcl_losses['bcl_attractive']

            # --- Binary semantic loss ---
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
        total_bcl_repulsive /= batch_size
        total_bcl_attractive /= batch_size

        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi,
            'bcl_repulsive': self.bcl_weight * total_bcl_repulsive,
            'bcl_attractive': self.bcl_weight * total_bcl_attractive,
        }

        # --- Second stage: query-based segmentation (same as parent) ---
        queries = []
        queries_inslabel = []
        queries_idx = []

        if self.prepare_epoch:
            if kwargs['epoch'] > self.prepare_epoch:
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

                        selected_indices_case4 = self._select_query_indices(
                            embed_logits[i], tree_indices,
                            embed_logits[i].device)

                        current_points = batch_inputs_dict['points'][i]
                        current_points_add = scatter_add(
                            current_points, voxel_superpoints, dim=0)
                        voxel_counts = scatter_add(
                            torch.ones_like(current_points[:, 0].float()),
                            voxel_superpoints, dim=0)
                        avg_points = current_points_add / \
                            voxel_counts.unsqueeze(-1).clamp(min=1)

                        queries.append(x[i][selected_indices_case4])

                        pts_instance_mask = \
                            batch_data_samples[i].gt_pts_seg.pts_instance_mask
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
                            (x[i], queries[i], batch_data_samples[i],
                             queries_inslabel[i], i)
                            for i in range(len(queries))
                            if len(queries[i]) > 0]
                        (x, queries, batch_data_samples,
                         queries_inslabel, original_indices) = \
                            zip(*filtered_results)
                        x = list(x)
                        queries = list(queries)
                        batch_data_samples = list(batch_data_samples)
                        queries_inslabel = list(queries_inslabel)
                        original_indices = list(original_indices)
                    else:
                        original_indices = list(
                            range(len(batch_data_samples)))

                    x = self.decoder(x, queries)

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

                        batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                            self.get_gt_semantic_masks(
                                sem_mask, voxel_superpoints, self.num_classes)
                        batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                            self.get_gt_inst_masks(
                                inst_mask, voxel_superpoints)

                        (batch_data_samples[i].gt_instances_3d.labels_3d,
                         batch_data_samples[i].gt_instances_3d.sp_inst_masks,
                         batch_data_samples[i].gt_instances_3d.ratio_inspoint
                         ) = self.filter_stuff_masks(
                            batch_data_samples[i].gt_instances_3d,
                            self.stuff_classes,
                            batch_data_samples[i].gt_pts_seg.ratio_inspoint)

                        batch_data_samples[i].gt_instances_3d.query_inslabel \
                            = queries_inslabel[i]
                        sp_gt_instances.append(
                            batch_data_samples[i].gt_instances_3d)

                    loss = self.criterion(x, sp_gt_instances)
                    loss_final.update(loss)
        return loss_final
