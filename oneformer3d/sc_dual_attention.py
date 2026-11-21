"""SCDualAtt: Spatial-Channel Dual Attention Module.

A lightweight parallel dual-attention block designed to enhance 3D sparse
U-Net backbone features before they enter the ISA MLP head and Transformer
Decoder. It operates on a SparseConvTensor and preserves sparsity.

Architecture (residual):

    F_in (N x C)
       |
       ├──► [Channel Branch] ──► M_c (1 x C) ──► F_c = M_c ⊙ F_in
       |
       ├──► [Spatial Branch] ──► M_s (N x 1) ──► F_s = M_s ⊙ F_in
       |
       └──► F_out = F_in + (F_c + F_s)

Channel Branch: GlobalAvgPool → MLP(C→C/r→C) → Sigmoid
Spatial Branch: Sparse 1×1×1 Conv → Sparse 3×3×3 Conv → Sigmoid
"""

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SCDualAtt(nn.Module):
    """Spatial-Channel Dual Attention for sparse 3D features.

    Insert this module **after** the 3D sparse U-Net backbone output and
    **before** the output BN+ReLU layer (i.e. inside ``extract_feat``).

    Args:
        in_channels (int): Number of input feature channels (C).
            Should match the U-Net output channel count.
        reduction (int): Channel compression ratio for the MLP bottleneck
            and the spatial 1×1 conv.  Defaults to 4.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)

        # -------- Channel Attention Branch --------
        # GAP → MLP (C → C/r → C) → Sigmoid
        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # -------- Spatial Attention Branch --------
        # Sparse 1×1×1 Conv (compress) → Sparse 3×3×3 Conv → Sigmoid
        self.spatial_compress = spconv.SubMConv3d(
            in_channels, mid_channels,
            kernel_size=1, bias=False)
        self.spatial_conv = spconv.SubMConv3d(
            mid_channels, 1,
            kernel_size=3, padding=1, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, sparse_tensor: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        """Forward pass.

        Args:
            sparse_tensor (SparseConvTensor): Input whose ``features``
                have shape (N, C) and ``indices`` have shape (N, 4).

        Returns:
            SparseConvTensor: Feature-enhanced sparse tensor with
                identical spatial structure (same indices / spatial_shape).
        """
        features = sparse_tensor.features  # (N, C)

        # ============ Channel Attention ============
        # Global average pooling across all N points
        gap = features.mean(dim=0, keepdim=True)          # (1, C)
        channel_weights = self.channel_sigmoid(
            self.channel_mlp(gap))                         # (1, C)
        f_channel = features * channel_weights             # (N, C)

        # ============ Spatial Attention ============
        # Sparse 1×1 → Sparse 3×3 → Sigmoid
        out = self.spatial_compress(sparse_tensor)         # SparseTensor (N, C/r)
        out = self.spatial_conv(out)                       # SparseTensor (N, 1)
        spatial_weights = self.spatial_sigmoid(
            out.features)                                  # (N, 1)
        f_spatial = features * spatial_weights             # (N, C)

        # ============ Fusion + Residual ============
        fused = f_channel + f_spatial                      # (N, C)
        enhanced = features + fused                        # residual

        return sparse_tensor.replace_feature(enhanced)
