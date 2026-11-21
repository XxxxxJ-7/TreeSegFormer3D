# SpConvUNet backbone enhanced with GFEM (Global-view Feature Encoding Module)
# from ITS-Net (Li et al., ISPRS J. 2026).
# Original SpConvUNet is NOT modified; this is a separate registered module.
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from spconv.pytorch.modules import SparseModule
from torch_cluster import knn

from mmdet3d.registry import MODELS
from .spconv_unet import ResidualBlock


class GFEM(nn.Module):
    """Global-view Feature Encoding Module (ITS-Net, Li et al. 2026).

    Enhances point/voxel features by computing channel-domain and
    spatial-domain heterogeneity attention from KNN neighborhoods.

    Args:
        in_channels (int): Number of input feature channels.
        mid_channels (int): Intermediate channels for attention heads.
            Kept small for memory efficiency.
        k (int): Number of nearest neighbours.
    """

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__()
        self.k = k
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        mlp_in = in_channels + 3
        self.psi = self._make_mlp(mlp_in, mid_channels)
        self.phi = self._make_mlp(mlp_in, mid_channels)
        self.alpha = self._make_mlp(mlp_in, mid_channels)
        self.beta = self._make_mlp(mlp_in, mid_channels)

        # Dual-branch attention: adaptive fusion of channel / spatial branches
        self.query_proj = nn.Sequential(
            nn.Linear(mlp_in, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.key_channel = nn.Linear(mid_channels, mid_channels)
        self.key_spatial = nn.Linear(mid_channels, mid_channels)
        self.attn_scale = mid_channels ** -0.5

        self.lbr = nn.Sequential(
            nn.Linear(mid_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _make_mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, sparse_tensor):
        """Forward pass operating on a SparseConvTensor.

        Args:
            sparse_tensor (SparseConvTensor): Input sparse tensor whose
                ``indices`` have shape (N, 4) = [batch, z, y, x] and
                ``features`` have shape (N, C).

        Returns:
            SparseConvTensor: Feature-enhanced sparse tensor (same sparsity).
        """
        features = sparse_tensor.features          # (N, C)
        indices = sparse_tensor.indices            # (N, 4)
        batch_idx = indices[:, 0].long()
        coords = indices[:, 1:].float()            # (N, 3)

        N = features.shape[0]
        K = min(self.k, N)
        if K < 2:
            return sparse_tensor

        # torch_cluster.knn requires inputs sorted by batch index
        order = batch_idx.argsort(stable=True)
        coords_s = coords[order]
        batch_s = batch_idx[order]
        feats_s = features[order]

        edge_index = knn(coords_s, coords_s, K,
                         batch_x=batch_s, batch_y=batch_s)
        row, col = edge_index[0], edge_index[1]

        n_edges = row.shape[0]
        expected = N * K
        if n_edges != expected:
            # Pad with self-loops for points that got fewer than K neighbours
            counts = torch.zeros(N, dtype=torch.long, device=row.device)
            counts.scatter_add_(0, row,
                                torch.ones_like(row, dtype=torch.long))
            deficit = K - counts                        # (N,)
            need_pad = deficit > 0
            if need_pad.any():
                pad_pts = need_pad.nonzero(as_tuple=True)[0]
                pad_row = pad_pts.repeat_interleave(deficit[pad_pts])
                pad_col = pad_row.clone()               # self-loop
                row = torch.cat([row, pad_row])
                col = torch.cat([col, pad_col])
                # re-sort so each point's K neighbours are contiguous
                sort_idx = row.argsort(stable=True)
                row = row[sort_idx]
                col = col[sort_idx]

        # relative position + neighbour features
        rel_pos = coords_s[col] - coords_s[row]        # (N*K, 3)
        nbr_feat = feats_s[col]                         # (N*K, C)
        x = torch.cat([rel_pos, nbr_feat], dim=-1)     # (N*K, 3+C)

        # four parallel transformations  ψ, φ, α, β
        psi_x = self.psi(x).view(N, K, -1)             # (N, K, D')
        phi_x = self.phi(x).view(N, K, -1)
        alpha_x = self.alpha(x).view(N, K, -1)
        beta_x = self.beta(x).view(N, K, -1)

        # --- channel-domain similarity  Sc = softmax(ψᵀφ) ---
        Sc = torch.softmax(
            torch.bmm(psi_x.transpose(1, 2), phi_x), dim=-1)

        # --- spatial-domain similarity  Ss = softmax(φψᵀ) ---
        Ss = torch.softmax(
            torch.bmm(phi_x, psi_x.transpose(1, 2)), dim=-1)

        # --- channel heterogeneity  Hc ---
        Sc_max = Sc.max(dim=-1, keepdim=True)[0].expand_as(Sc)
        Hc = torch.softmax(Sc_max - Sc, dim=-1)

        # --- spatial heterogeneity  Hs ---
        Ss_max = Ss.max(dim=-1, keepdim=True)[0].expand_as(Ss)
        Hs = torch.softmax(Ss_max - Ss, dim=-1)

        # --- two branch outputs ---
        beta_hc = torch.bmm(beta_x, Hc)                # (N, K, D')
        hs_alpha = torch.bmm(Hs, alpha_x)              # (N, K, D')

        channel_feat = beta_hc.max(dim=1)[0]            # (N, D')
        spatial_feat = hs_alpha.max(dim=1)[0]           # (N, D')

        # --- dual-branch attention: input-dependent adaptive fusion ---
        query = self.query_proj(
            torch.cat([coords_s, feats_s], dim=-1))     # (N, D')
        key_c = self.key_channel(channel_feat)          # (N, D')
        key_s = self.key_spatial(spatial_feat)           # (N, D')

        score_c = (query * key_c).sum(dim=-1, keepdim=True) * self.attn_scale
        score_s = (query * key_s).sum(dim=-1, keepdim=True) * self.attn_scale
        attn = torch.softmax(
            torch.cat([score_c, score_s], dim=-1), dim=-1)  # (N, 2)

        fused = (attn[:, 0:1] * channel_feat
                 + attn[:, 1:2] * spatial_feat)         # (N, D')
        fused = self.lbr(fused)                         # (N, C)

        enhanced_s = feats_s + fused                    # residual (sorted)

        # map back to original voxel order
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(N, device=order.device)
        enhanced = enhanced_s[inv_order]

        return sparse_tensor.replace_feature(enhanced)


@MODELS.register_module()
class SpConvUNet_GFEM(nn.Module):
    """SpConvUNet backbone enhanced with GFEM at selected encoder levels.

    The architecture is identical to SpConvUNet except that a GFEM block
    is inserted after the encoder residual blocks (before down-sampling).

    Args:
        num_planes (List[int]): Channels per level.
        norm_fn: Normalisation constructor.
        block_reps (int): Repetitions of residual blocks.
        block: Block class.
        indice_key_id (int): SpConv indice key counter.
        normalize_before (bool): BN-ReLU before conv.
        return_blocks (bool): Return intermediate block outputs.
        gfem_k (int): KNN neighbours for GFEM.
        gfem_mid_channels (int): Attention dimension in GFEM.
        gfem_levels (list[int] | int | None): Which levels to enable GFEM
            (1-indexed). Accepts:
              - a list, e.g. [2, 3] → only level 2 and 3;
              - an int  (start_level), e.g. 3 → levels 3,4,5,…;
              - None → all levels.
    """

    def __init__(self,
                 num_planes,
                 norm_fn=functools.partial(
                     nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2,
                 block=ResidualBlock,
                 indice_key_id=1,
                 normalize_before=True,
                 return_blocks=False,
                 gfem_k=16,
                 gfem_mid_channels=32,
                 gfem_levels=None,
                 gfem_start_level=None):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        # ---------- encoder residual blocks ----------
        blocks = {
            f'block{i}': block(
                num_planes[0], num_planes[0], norm_fn,
                normalize_before=normalize_before,
                indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        self.blocks = spconv.SparseSequential(OrderedDict(blocks))

        # ---------- GFEM (enabled on selected levels) ----------
        # Normalise the three config styles into a set of active levels
        if gfem_levels is not None:
            # explicit list, e.g. [2, 3]
            self._gfem_level_set = set(gfem_levels)
        elif gfem_start_level is not None:
            # legacy: enable from start_level onward
            self._gfem_level_set = set(range(gfem_start_level,
                                             len(num_planes) + indice_key_id + 1))
        else:
            # default: all levels
            self._gfem_level_set = None  # means "all"

        self.use_gfem = (self._gfem_level_set is None
                         or indice_key_id in self._gfem_level_set)
        if self.use_gfem:
            self.gfem = GFEM(num_planes[0], mid_channels=gfem_mid_channels, k=gfem_k)

        # ---------- deeper levels ----------
        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0], num_planes[1],
                        kernel_size=2, stride=2, bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0], num_planes[1],
                        kernel_size=2, stride=2, bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[1]), nn.ReLU())

            self.u = SpConvUNet_GFEM(
                num_planes[1:], norm_fn, block_reps, block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                gfem_k=gfem_k,
                gfem_mid_channels=gfem_mid_channels,
                gfem_levels=gfem_levels,
                gfem_start_level=gfem_start_level)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1], num_planes[0],
                        kernel_size=2, bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1], num_planes[0],
                        kernel_size=2, bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    num_planes[0] * (2 - i), num_planes[0], norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            self.blocks_tail = spconv.SparseSequential(
                OrderedDict(blocks_tail))

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input sparse tensor.
            previous_outputs (List[SparseConvTensor]): Intermediate outputs.

        Returns:
            SparseConvTensor (or tuple): Output sparse tensor.
        """
        output = self.blocks(input)

        # GFEM enhancement after encoder blocks (if enabled at this level)
        if self.use_gfem:
            output = self.gfem(output)

        identity = spconv.SparseConvTensor(
            output.features, output.indices,
            output.spatial_shape, output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features),
                           dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output
