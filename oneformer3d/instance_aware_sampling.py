"""Instance-Aware Query Sampling for ForAINetV2OneFormer3D.

Replaces vanilla FPS query selection with a two-stage strategy:
  1. Cluster foreground embeddings to discover potential instances
  2. Sample queries per-cluster to guarantee every instance gets coverage

Only the query selection logic changes; backbone, decoder, loss criterion,
and predict post-processing are all inherited unchanged.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
import os
import time
import gc

from torch_scatter import scatter_add
from torch_cluster import fps
from tqdm import tqdm

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData

from .oneformer3d import ForAINetV2OneFormer3D_XAwarequery
from .panoptic_losses import discriminative_loss


def instance_aware_fps(embed_logits, tree_indices, query_point_num,
                       device, n_seeds=50, min_per_cluster=3):
    """Instance-aware query selection.

    Instead of a single global FPS, first discover potential instance
    clusters in the embedding space, then sample a balanced number of
    queries from each cluster.

    Args:
        embed_logits (Tensor): Embedding features for ALL voxels,
            shape ``(N_voxels, embed_dim)``.
        tree_indices (Tensor): Indices into ``embed_logits`` that
            correspond to foreground (tree) voxels.
        query_point_num (int): Total number of queries to select.
        device: Torch device.
        n_seeds (int): Number of seed points for clustering.
        min_per_cluster (int): Minimum queries per cluster.

    Returns:
        Tensor: Selected indices (into the full voxel array).
    """
    tree_embeds = embed_logits[tree_indices]      # (N_tree, D)
    n_tree = tree_embeds.size(0)

    if n_tree <= query_point_num:
        return tree_indices

    # --- Stage 1: discover clusters via seed-based assignment ---
    n_seeds_actual = min(n_seeds, n_tree // 2, n_tree)
    if n_seeds_actual < 2:
        batch_t = torch.zeros(n_tree, dtype=torch.long, device=device)
        ratio = min(query_point_num / n_tree,
                    torch.tensor(1.0, device=device))
        sel = fps(tree_embeds, batch_t, ratio=ratio)
        return tree_indices[sel[:query_point_num]]

    batch_t = torch.zeros(n_tree, dtype=torch.long, device=device)
    seed_ratio = min(n_seeds_actual / n_tree,
                     torch.tensor(1.0, device=device))
    seed_local = fps(tree_embeds, batch_t, ratio=seed_ratio)
    seed_feats = tree_embeds[seed_local]                  # (S, D)

    # assign every tree voxel to its nearest seed
    # process in chunks to avoid OOM on large point clouds
    chunk_size = 20000
    cluster_ids = torch.empty(n_tree, dtype=torch.long, device=device)
    for start in range(0, n_tree, chunk_size):
        end = min(start + chunk_size, n_tree)
        dists = torch.cdist(tree_embeds[start:end], seed_feats)  # (chunk, S)
        cluster_ids[start:end] = dists.argmin(dim=1)

    # --- Stage 2: merge tiny clusters only if they are close in embedding ---
    # Tiny clusters (< min_per_cluster voxels) are merged into the nearest
    # large cluster ONLY when the embedding distance is small.  Otherwise
    # they are kept as independent clusters — this protects genuinely
    # small trees from being swallowed by nearby large trees.
    unique_cids, counts = cluster_ids.unique(return_counts=True)
    tiny_mask = counts < min_per_cluster
    if tiny_mask.any():
        large_cids = unique_cids[~tiny_mask]
        if large_cids.numel() > 0:
            all_centers = torch.stack(
                [tree_embeds[cluster_ids == c].mean(0) for c in unique_cids])
            large_centers = torch.stack(
                [tree_embeds[cluster_ids == c].mean(0) for c in large_cids])
            # adaptive threshold: median inter-cluster distance * 0.5
            all_dists = torch.cdist(all_centers, all_centers)
            all_dists.fill_diagonal_(float('inf'))
            merge_threshold = all_dists.min(dim=1).values.median() * 0.5
            for tc in unique_cids[tiny_mask]:
                tc_center = tree_embeds[cluster_ids == tc].mean(
                    0, keepdim=True)
                dists_to_large = torch.cdist(tc_center, large_centers)
                min_dist, nearest = dists_to_large.min(dim=1)
                if min_dist.item() < merge_threshold.item():
                    cluster_ids[cluster_ids == tc] = \
                        large_cids[nearest.item()]
                # else: keep tiny cluster as-is (likely a small tree)

    # --- Stage 3: balanced sampling from each cluster ---
    unique_cids = cluster_ids.unique()
    n_clusters = unique_cids.numel()
    base_budget = max(min_per_cluster, query_point_num // n_clusters)

    all_selected = []
    for cid in unique_cids:
        mask = cluster_ids == cid
        n_in_cluster = mask.sum().item()
        n_sample = min(base_budget, n_in_cluster)

        if n_sample >= n_in_cluster:
            local_sel = torch.where(mask)[0]
        else:
            cluster_embeds = tree_embeds[mask]
            b = torch.zeros(n_in_cluster, dtype=torch.long, device=device)
            ratio = min(n_sample / n_in_cluster,
                        torch.tensor(1.0, device=device))
            fps_idx = fps(cluster_embeds, b, ratio=ratio)
            fps_idx = fps_idx[:n_sample]
            local_sel = torch.where(mask)[0][fps_idx]
        all_selected.append(local_sel)

    selected = torch.cat(all_selected)

    # --- Stage 4: adjust total count ---
    if selected.numel() > query_point_num:
        perm = torch.randperm(selected.numel(), device=device)[:query_point_num]
        selected = selected[perm]
    elif selected.numel() < query_point_num:
        remaining = query_point_num - selected.numel()
        already = set(selected.tolist())
        pool = torch.tensor(
            [i for i in range(n_tree) if i not in already],
            dtype=torch.long, device=device)
        if pool.numel() > 0:
            pool_embeds = tree_embeds[pool]
            b = torch.zeros(pool.numel(), dtype=torch.long, device=device)
            ratio = min(remaining / pool.numel(),
                        torch.tensor(1.0, device=device))
            extra = fps(pool_embeds, b, ratio=ratio)
            extra = extra[:remaining]
            selected = torch.cat([selected, pool[extra]])

    return tree_indices[selected[:query_point_num]]


# =========================================================================
#  Model with Instance-Aware Sampling
# =========================================================================

@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_IAS(
        ForAINetV2OneFormer3D_XAwarequery):
    """ForAINetV2OneFormer3D_XAwarequery with instance-aware query sampling.

    Overrides ``loss`` and ``predict`` to replace vanilla FPS with
    ``instance_aware_fps``, ensuring every potential instance gets
    query coverage.  Everything else is inherited unchanged.

    Extra config keys:
        n_seeds (int): Number of clustering seeds. Default 50.
        min_per_cluster (int): Minimum queries per cluster. Default 3.
    """

    def __init__(self, n_seeds=50, min_per_cluster=3, **kwargs):
        super().__init__(**kwargs)
        self.n_seeds = n_seeds
        self.min_per_cluster = min_per_cluster

    def _select_query_indices(self, embed_logits_i, tree_indices, device):
        """Select query voxel indices with instance-aware sampling."""
        return instance_aware_fps(
            embed_logits_i, tree_indices, self.query_point_num,
            device, n_seeds=self.n_seeds,
            min_per_cluster=self.min_per_cluster)

    # ------------------------------------------------------------------
    #  loss  – identical to parent except FPS → _select_query_indices
    # ------------------------------------------------------------------

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
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

                        # === INSTANCE-AWARE SAMPLING (replaces vanilla FPS) ===
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

    # ------------------------------------------------------------------
    #  predict  – identical to parent except FPS → _select_query_indices
    # ------------------------------------------------------------------

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        t0 = time.time()
        lidar_path = batch_data_samples[0].lidar_path
        base_name = os.path.basename(lidar_path)
        current_filename = os.path.splitext(base_name)[0]
        t1 = time.time()

        if 'test' in lidar_path:
            step_size = self.radius / 4
            grid_size = 0.2
            num_points = 640000
            pts_semantic_gt = \
                batch_data_samples[0].eval_ann_info['pts_semantic_mask']
            pts_instance_gt = \
                batch_data_samples[0].eval_ann_info['pts_instance_mask']
            original_points = batch_inputs_dict['points'][0]
            regions = self.generate_cylindrical_regions(
                original_points, self.radius, step_size)
            num_cls = self.test_cfg.num_sem_cls
            votes_counter = np.zeros(
                (original_points.shape[0], num_cls), dtype=np.int16)
            all_pre_ins = np.full(original_points.shape[0], -1)
            max_instance = 0

            global_instance_scores = np.zeros(
                (original_points.shape[0],), dtype=float)
            best_masks = []
            all_instance_labels = set(np.unique(pts_instance_gt))

            output_path = self.test_cfg.get(
                'output_dir', 'work_dirs/default_output')
            score_th1 = self.score_th
            score_th2 = 0.3
            t2 = time.time()
            last_results = None
            last_originids = None

            for region_idx, region in enumerate(
                    tqdm(regions, desc="Processing regions")):
                t2 = time.time()
                region_mask = (
                    (original_points[:, 0] - region[0]) ** 2 +
                    (original_points[:, 1] - region[1]) ** 2
                ) <= self.radius ** 2
                pc1 = original_points[region_mask]
                pc1_indices = torch.where(region_mask)[0]
                t3 = time.time()
                if len(pc1) == 0:
                    continue

                pc2, pc2_indices = self.grid_sample(
                    pc1, pc1_indices, grid_size)
                if len(pc2) < num_points:
                    pc3 = pc2
                    pc3_indices = pc2_indices
                elif len(pc2) > num_points:
                    pc3, pc3_indices = self.points_random_sampling(
                        pc2, pc2_indices, num_points)

                coordinates, features, inverse_mapping2, spatial_shape = \
                    self.collate([pc3])
                x = spconv.SparseConvTensor(
                    features, coordinates, spatial_shape,
                    len(batch_data_samples))
                x = self.extract_feat(x)
                embed_logits = self.Embed(x[0])
                bi_semantic_logits = self.BiSemantic(x[0])

                wood_class = 1
                semantic_predictions_bi = torch.argmax(
                    bi_semantic_logits, dim=1)
                tree_indices = torch.where(
                    semantic_predictions_bi == wood_class)[0]

                with torch.no_grad():
                    nn_idx_pc1 = []
                    chunk = self.chunk
                    for ss in range(0, pc1.shape[0], chunk):
                        ee = min(ss + chunk, pc1.shape[0])
                        nn_idx_pc1.append(
                            torch.cdist(
                                pc1[ss:ee].float(), pc3.float()
                            ).argmin(1))
                    nn_idx_pc1 = torch.cat(nn_idx_pc1)

                if tree_indices.numel() > 1:
                    # === INSTANCE-AWARE SAMPLING (replaces vanilla FPS) ===
                    selected_indices_case4 = self._select_query_indices(
                        embed_logits, tree_indices, embed_logits.device)

                    queries = []
                    queries.append(x[0][selected_indices_case4])
                    x = self.decoder(x, queries)
                    results_list = self.predict_by_feat_test(
                        x, inverse_mapping2, pc3, selected_indices_case4)

                    masks = results_list[0].pts_instance_mask[0]
                    scores = results_list[0].instance_scores
                    valid_scores_mask = scores > score_th1

                    center = region
                    edge_threshold = self.radius - 0.5
                    pc3_distances = torch.sqrt(
                        (pc3[:, 0] - center[0]) ** 2 +
                        (pc3[:, 1] - center[1]) ** 2)

                    valid_scores_mask2 = np.ones_like(
                        valid_scores_mask, dtype=bool)
                    for i, mask in enumerate(masks):
                        if valid_scores_mask[i]:
                            if torch.any(
                                    pc3_distances[mask] > edge_threshold):
                                valid_scores_mask2[i] = False

                    keep = torch.where(
                        torch.tensor(
                            valid_scores_mask, device=pc3.device) &
                        torch.tensor(
                            valid_scores_mask2, device=pc3.device)
                    )[0]

                    if keep.numel():
                        masks_kept = torch.as_tensor(
                            masks, device=pc3.device)[keep]
                        scores_kept = torch.as_tensor(
                            scores, device=pc3.device)[keep]

                        rows_list, cols_list = [], []
                        max_chunk = (
                            torch.iinfo(torch.int32).max //
                            masks_kept.shape[0]) - 1_000_000
                        max_chunk = max(1, max_chunk)
                        chunk_size = min(
                            nn_idx_pc1.shape[0], max_chunk)

                        for start in range(
                                0, nn_idx_pc1.shape[0], chunk_size):
                            end = min(
                                start + chunk_size, nn_idx_pc1.shape[0])
                            mk_bool_chunk = masks_kept[
                                :, nn_idx_pc1[start:end]]
                            rows_chunk, cols_chunk = \
                                mk_bool_chunk.nonzero(as_tuple=True)
                            cols_chunk = cols_chunk + start
                            rows_list.append(rows_chunk)
                            cols_list.append(cols_chunk)

                        rows = torch.cat(rows_list, dim=0)
                        cols = torch.cat(cols_list, dim=0)
                        score_per_hit = scores_kept[rows]

                        N1 = pc1.shape[0]
                        best_score = torch.full(
                            (N1,), -1., device=pc3.device)
                        best_mid = torch.full(
                            (N1,), -1, dtype=torch.long,
                            device=pc3.device)

                        best_score.index_reduce_(
                            0, cols, score_per_hit, reduce='amax')
                        improved_mask = \
                            score_per_hit == best_score[cols]
                        best_mid.index_put_(
                            (cols[improved_mask],),
                            rows[improved_mask], accumulate=False)

                        pts_glob = pc1_indices.cpu().numpy()
                        new_scores = best_score.cpu().numpy()
                        better_pts = (
                            new_scores >
                            global_instance_scores[pts_glob])

                        if better_pts.any():
                            global_instance_scores[
                                pts_glob[better_pts]] = \
                                new_scores[better_pts]
                            all_pre_ins[pts_glob[better_pts]] = (
                                max_instance +
                                best_mid.cpu().numpy()[better_pts])

                            mids_unique = np.unique(
                                best_mid.cpu().numpy()[better_pts])
                            for mid in mids_unique:
                                sel_pts = (
                                    cols[rows == mid]).cpu().numpy()
                                best_masks.append((
                                    pts_glob[sel_pts],
                                    max_instance + int(mid),
                                    float(scores_kept[int(mid)])))

                        max_instance += int(masks_kept.size(0))

                    sem_pred_pc3 = results_list[0].pts_semantic_mask[0]
                    if isinstance(sem_pred_pc3, np.ndarray):
                        sem_pred_pc3 = torch.from_numpy(
                            sem_pred_pc3).to(pc3.device)
                    else:
                        sem_pred_pc3 = sem_pred_pc3.to(pc3.device)
                    cylinder_current_semantic_pre = \
                        sem_pred_pc3[nn_idx_pc1]

                    ids_np = torch.where(
                        region_mask)[0].cpu().numpy()
                    sem_np = (cylinder_current_semantic_pre
                              .cpu().numpy().astype(int))
                    np.add.at(votes_counter, (ids_np, sem_np), 1)

                    originids = pc3_indices.cpu().numpy()
                    last_results = results_list
                    last_originids = originids
                    last_batch_len = len(batch_data_samples)

                else:
                    projected_semantic_logits = \
                        bi_semantic_logits[inverse_mapping2]
                    proj_logits = bi_semantic_logits[inverse_mapping2]
                    sem_pred_pc3 = torch.argmax(proj_logits, dim=1)
                    if isinstance(sem_pred_pc3, np.ndarray):
                        sem_pred_pc3 = torch.from_numpy(
                            sem_pred_pc3).to(pc3.device)
                    else:
                        sem_pred_pc3 = sem_pred_pc3.to(pc3.device)
                    cylinder_current_semantic_pre = \
                        sem_pred_pc3[nn_idx_pc1]

                    ids_np = torch.where(
                        region_mask)[0].cpu().numpy()
                    sem_np = (cylinder_current_semantic_pre
                              .cpu().numpy().astype(int))
                    np.add.at(votes_counter, (ids_np, sem_np), 1)

                del pc1, pc1_indices, pc2, pc2_indices, pc3, pc3_indices
                del embed_logits, bi_semantic_logits
                torch.cuda.empty_cache()
                gc.collect()

            final_semantic_labels = votes_counter.argmax(1)
            final_semantic_labels[votes_counter.sum(1) == 0] = -1
            ground_mask = (final_semantic_labels == 0)
            all_pre_ins[ground_mask] = -1

            t11 = time.time()
            uniq, cnt = np.unique(all_pre_ins, return_counts=True)
            to_kill = np.isin(
                all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])
            all_pre_ins[to_kill] = -1
            t12 = time.time()
            print(f"postprocessing 6: {(t12 - t11)*1000:.0f} ms")

            unique_best_masks = []
            for mask_points, instance_id, score in best_masks:
                if np.any(all_pre_ins[mask_points] == instance_id):
                    unique_best_masks.append(
                        (mask_points, instance_id, score))
            t13 = time.time()
            print(f"postprocessing 7: {(t13 - t12)*1000:.0f} ms")

            clean_all_pre_ins, merged_masks, merged_instance_scores = \
                self.merge_overlapping_instances_by_score_speedup(
                    all_pre_ins, unique_best_masks,
                    overlap_threshold=score_th2)
            t14 = time.time()
            print(f"postprocessing 8: {(t14 - t13)*1000:.0f} ms")

            unique_labels = np.unique(clean_all_pre_ins)
            unique_labels = unique_labels[unique_labels >= 0]
            relabel_map = {
                old_label: new_label
                for new_label, old_label in enumerate(unique_labels)}
            relabel_map[-1] = -1
            clean_all_pre_ins = np.vectorize(
                relabel_map.get)(clean_all_pre_ins)

            region_path = os.path.join(
                output_path, f"{current_filename}.ply")
            self.save_ply_withscore(
                original_points.cpu().numpy(),
                final_semantic_labels, clean_all_pre_ins,
                merged_instance_scores, region_path,
                pts_semantic_gt, pts_instance_gt)

            if (last_results is not None and
                    len(last_results) == len(batch_data_samples)):
                for i, data_sample in enumerate(batch_data_samples):
                    data_sample.pred_pts_seg = last_results[i]
                    data_sample.pred_pts_seg['originids'] = \
                        last_originids
            else:
                for data_sample in batch_data_samples:
                    data_sample.pred_pts_seg = None
            return batch_data_samples
        else:
            coordinates, features, inverse_mapping, spatial_shape = \
                self.collate(batch_inputs_dict['points'])
            x = spconv.SparseConvTensor(
                features, coordinates, spatial_shape,
                len(batch_data_samples))
            x = self.extract_feat(x)

            embed_logits = [self.Embed(y) for y in x]
            bi_semantic_logits = [self.BiSemantic(y) for y in x]

            queries = []
            for i in range(len(x)):
                wood_class = 1
                semantic_predictions_bi = torch.argmax(
                    bi_semantic_logits[i], dim=1)
                tree_indices = torch.where(
                    semantic_predictions_bi == wood_class)[0]

                if tree_indices.numel() > 1:
                    selected_indices = self._select_query_indices(
                        embed_logits[i], tree_indices,
                        embed_logits[i].device)
                    queries.append(x[i][selected_indices])
                else:
                    max_len = min(self.query_point_num, len(x[i]))
                    queries.append(x[i][0:max_len])

            x = self.decoder(x, queries)
            results_list = self.predict_by_feat(x, inverse_mapping)

            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_pts_seg = results_list[i]

            return batch_data_samples
