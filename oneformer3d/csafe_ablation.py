# =============================================================================
# CSAFE – Fusion mechanism ablation variants
# =============================================================================
# Every variant shares the same KNN → psi/phi/alpha/beta → Hc/Hs two-branch
# encoding.  ONLY the fusion logic differs.  All classes are registered with
# MMDetection so they can be swapped via config.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn

from mmdet3d.registry import MODELS


# ---------------------------------------------------------------------------
#  Shared base class — contains psi/phi/alpha/beta, KNN, Hc/Hs computation
# ---------------------------------------------------------------------------
class _TwoBranchBase(nn.Module):
    """Common two-branch encoding: KNN → psi/phi/alpha/beta → Hc/Hs.

    Subclasses MUST override ``_fuse(channel_feat, spatial_feat, coords_s,
    feats_s)`` and return ``(fused: (N, D'), out_features: (N, C))``.
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

        # output projection shared across all variants
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

    # ---- two-branch encoding (identical for all variants) ----
    def _compute_branches(self, coords_s, feats_s, N, K):
        """Returns channel_feat (N,D'), spatial_feat (N,D')."""
        # KNN graph
        batch_s = torch.zeros(N, dtype=torch.long, device=coords_s.device)
        edge_index = knn(coords_s, coords_s, K,
                         batch_x=batch_s, batch_y=batch_s)
        row, col = edge_index[0], edge_index[1]

        # pad short edges
        n_edges = row.shape[0]
        expected = N * K
        if n_edges != expected:
            counts = torch.zeros(N, dtype=torch.long, device=row.device)
            counts.scatter_add_(0, row,
                                torch.ones_like(row, dtype=torch.long))
            deficit = K - counts
            need_pad = deficit > 0
            if need_pad.any():
                pad_pts = need_pad.nonzero(as_tuple=True)[0]
                pad_row = pad_pts.repeat_interleave(deficit[pad_pts])
                pad_col = pad_row.clone()
                row = torch.cat([row, pad_row])
                col = torch.cat([col, pad_col])
                sort_idx = row.argsort(stable=True)
                row = row[sort_idx]
                col = col[sort_idx]

        # edge features
        rel_pos = coords_s[col] - coords_s[row]       # (N*K, 3)
        nbr_feat = feats_s[col]                         # (N*K, C)
        x = torch.cat([rel_pos, nbr_feat], dim=-1)     # (N*K, 3+C)

        # four parallel MLPs
        psi_x = self.psi(x).view(N, K, -1)             # (N, K, D')
        phi_x = self.phi(x).view(N, K, -1)
        alpha_x = self.alpha(x).view(N, K, -1)
        beta_x = self.beta(x).view(N, K, -1)

        # channel-domain similarity & heterogeneity
        Sc = torch.softmax(
            torch.bmm(psi_x.transpose(1, 2), phi_x), dim=-1)
        Hc = torch.softmax(
            Sc.max(dim=-1, keepdim=True)[0].expand_as(Sc) - Sc, dim=-1)

        # spatial-domain similarity & heterogeneity
        Ss = torch.softmax(
            torch.bmm(phi_x, psi_x.transpose(1, 2)), dim=-1)
        Hs = torch.softmax(
            Ss.max(dim=-1, keepdim=True)[0].expand_as(Ss) - Ss, dim=-1)

        # two-branch outputs
        channel_feat = torch.bmm(beta_x, Hc).max(dim=1)[0]    # (N, D')
        spatial_feat = torch.bmm(Hs, alpha_x).max(dim=1)[0]   # (N, D')

        return channel_feat, spatial_feat

    # ---- fusion (to be overridden) ----
    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        """Return (fused_D: (N,mid_channels), out: (N,in_channels))."""
        raise NotImplementedError

    # ---- full forward ----
    def forward(self, sparse_tensor):
        import spconv.pytorch as spconv

        features = sparse_tensor.features
        indices = sparse_tensor.indices
        batch_idx = indices[:, 0].long()
        coords = indices[:, 1:].float()

        N = features.shape[0]
        K = min(self.k, N)
        if K < 2:
            return sparse_tensor

        order = batch_idx.argsort(stable=True)
        coords_s = coords[order]
        feats_s = features[order]

        channel_feat, spatial_feat = self._compute_branches(
            coords_s, feats_s, N, K)

        # ---------------  variant-specific fusion  ---------------
        _, out_feat = self._fuse(channel_feat, spatial_feat, coords_s, feats_s)
        # ----------------------------------------------------------

        enhanced_s = feats_s + out_feat                    # residual

        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(N, device=order.device)
        enhanced = enhanced_s[inv_order]

        return sparse_tensor.replace_feature(enhanced)


# ===========================================================================
#  (a)  Simple sum  –  fused = channel_feat + spatial_feat
# ===========================================================================
@MODELS.register_module()
class CSAFE_SumFusion(_TwoBranchBase):
    """Ablation (a): fixed 1:1 sum of the two branches."""

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        fused = channel_feat + spatial_feat
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (b)  Concat + linear projection
# ===========================================================================
@MODELS.register_module()
class CSAFE_ConcatProj(_TwoBranchBase):
    """Ablation (b): concat → Linear(dim*2 → dim) – global static mapping."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        self.concat_proj = nn.Linear(mid_channels * 2, mid_channels)

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        fused = self.concat_proj(torch.cat([channel_feat, spatial_feat], dim=-1))
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (x)  Original GFEM – per-channel learnable weights, then sum
#       fused = W_c ⊙ channel_feat + W_s ⊙ spatial_feat
#       where W_c, W_s ∈ ℝ^D' are trainable vectors (input-agnostic,
#       but each channel has its own weight, unlike the scalar variant).
# ===========================================================================
@MODELS.register_module()
class CSAFE_GFEMOriginal(_TwoBranchBase):
    """Original GFEM (ITS-Net): per-channel learnable weights, then sum."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        self.w_channel = nn.Parameter(torch.ones(mid_channels))
        self.w_spatial = nn.Parameter(torch.ones(mid_channels))

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        fused = self.w_channel * channel_feat + self.w_spatial * spatial_feat
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (c)  Learnable scalar weights  (input-agnostic, coarser than original GFEM)
# ===========================================================================
@MODELS.register_module()
class CSAFE_LearnableScalar(_TwoBranchBase):
    """Ablation (c): two trainable scalars shared across ALL points/scenes."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        self.w_c = nn.Parameter(torch.tensor(1.0))
        self.w_s = nn.Parameter(torch.tensor(1.0))

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        # softmax so w_c + w_s = 1
        weights = F.softmax(torch.stack([self.w_c, self.w_s]), dim=0)
        fused = weights[0] * channel_feat + weights[1] * spatial_feat
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (d)  Key projection only  (no query_proj → pure linear alignment)
# ===========================================================================
@MODELS.register_module()
class CSAFE_NoQueryProj(_TwoBranchBase):
    """Ablation (d): key projections kept; query = Linear([P;F]) w/o BN+ReLU."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        mlp_in = in_channels + 3
        # plain linear for dimension alignment only
        self.query_align = nn.Linear(mlp_in, mid_channels)
        self.key_channel = nn.Linear(mid_channels, mid_channels)
        self.key_spatial = nn.Linear(mid_channels, mid_channels)
        self.attn_scale = mid_channels ** -0.5

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        query = self.query_align(torch.cat([coords_s, feats_s], dim=-1))
        key_c = self.key_channel(channel_feat)
        key_s = self.key_spatial(spatial_feat)

        score_c = (query * key_c).sum(dim=-1, keepdim=True) * self.attn_scale
        score_s = (query * key_s).sum(dim=-1, keepdim=True) * self.attn_scale
        attn = torch.softmax(
            torch.cat([score_c, score_s], dim=-1), dim=-1)

        fused = attn[:, 0:1] * channel_feat + attn[:, 1:2] * spatial_feat
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (e)  Query projection only  (no key projections)
# ===========================================================================
@MODELS.register_module()
class CSAFE_NoKeyProj(_TwoBranchBase):
    """Ablation (e): query_proj kept; keys = raw branch outputs."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        mlp_in = in_channels + 3
        self.query_proj = nn.Sequential(
            nn.Linear(mlp_in, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.attn_scale = mid_channels ** -0.5

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        query = self.query_proj(torch.cat([coords_s, feats_s], dim=-1))
        # keys are the raw branch outputs (no linear transform)
        key_c = channel_feat
        key_s = spatial_feat

        score_c = (query * key_c).sum(dim=-1, keepdim=True) * self.attn_scale
        score_s = (query * key_s).sum(dim=-1, keepdim=True) * self.attn_scale
        attn = torch.softmax(
            torch.cat([score_c, score_s], dim=-1), dim=-1)

        fused = attn[:, 0:1] * channel_feat + attn[:, 1:2] * spatial_feat
        out = self.lbr(fused)
        return fused, out


# ===========================================================================
#  (f)  Full CSAFE  – query_proj + key_channel + key_spatial
# ===========================================================================
@MODELS.register_module()
class CSAFE(_TwoBranchBase):
    """Full CSAFE: query_proj(BN+ReLU) + dual key projections + adaptive fusion."""

    def __init__(self, in_channels, mid_channels=32, k=16):
        super().__init__(in_channels, mid_channels, k)
        mlp_in = in_channels + 3
        self.query_proj = nn.Sequential(
            nn.Linear(mlp_in, mid_channels),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.key_channel = nn.Linear(mid_channels, mid_channels)
        self.key_spatial = nn.Linear(mid_channels, mid_channels)
        self.attn_scale = mid_channels ** -0.5

    def _fuse(self, channel_feat, spatial_feat, coords_s, feats_s):
        query = self.query_proj(torch.cat([coords_s, feats_s], dim=-1))
        key_c = self.key_channel(channel_feat)
        key_s = self.key_spatial(spatial_feat)

        score_c = (query * key_c).sum(dim=-1, keepdim=True) * self.attn_scale
        score_s = (query * key_s).sum(dim=-1, keepdim=True) * self.attn_scale
        attn = torch.softmax(
            torch.cat([score_c, score_s], dim=-1), dim=-1)

        fused = attn[:, 0:1] * channel_feat + attn[:, 1:2] * spatial_feat
        out = self.lbr(fused)
        return fused, out
