"""SpConvUNet with sparse window self-attention.

Adds window-based multi-head self-attention with 3-D relative position
bias at configurable levels of the recursive U-Net.  The attention block
is inserted after the encoder residual blocks so that enriched features
flow both through the skip connection and into deeper levels.
"""
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from torch import nn

from mmdet3d.registry import MODELS
from .spconv_unet import ResidualBlock


# --------------------------------------------------------------------------- #
#  Sparse Window Self-Attention
# --------------------------------------------------------------------------- #
class SparseWindowAttention(nn.Module):
    """Window-based multi-head self-attention for sparse voxel tensors.

    The 3-D voxel grid is partitioned into non-overlapping cubic windows of
    side ``window_size``.  Standard multi-head self-attention (with learnable
    3-D relative position bias) is applied independently within each window.

    A pre-LayerNorm residual connection wraps the whole block.

    Args:
        channels (int): Feature dimensionality.
        num_heads (int): Number of attention heads.
        window_size (int): Side length of each cubic window in voxels.
        attn_drop (float): Dropout rate on attention weights.
        proj_drop (float): Dropout rate on the output projection.
    """

    def __init__(self, channels, num_heads=4, window_size=8,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert channels % num_heads == 0, \
            f'channels ({channels}) must be divisible by num_heads ({num_heads})'
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        num_rel = (2 * window_size - 1) ** 3
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_rel, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        """Forward pass.

        Args:
            x (SparseConvTensor): Input sparse tensor.

        Returns:
            SparseConvTensor: Output with the same indices / spatial_shape.
        """
        feats = x.features                          # (N, C)
        coords = x.indices                          # (N, 4) [batch, z, y, x]
        N, C = feats.shape
        H, D = self.num_heads, self.head_dim
        ws = self.window_size
        device = feats.device

        identity = feats
        feats = self.norm(feats)

        # --- assign each voxel to a window -------------------------------- #
        batch_ids = coords[:, 0]
        xyz = coords[:, 1:]                         # (N, 3)
        win_xyz = xyz // ws                         # window grid coords
        local_xyz = xyz - win_xyz * ws              # [0, ws)

        sp = [int(s) for s in x.spatial_shape]
        gd = [(sp[i] + ws - 1) // ws for i in range(3)]
        batch_stride = gd[0] * gd[1] * gd[2]
        win_ids = (batch_ids * batch_stride
                   + win_xyz[:, 0] * (gd[1] * gd[2])
                   + win_xyz[:, 1] * gd[2]
                   + win_xyz[:, 2])

        unique_wins, inverse, counts = win_ids.unique(
            return_inverse=True, return_counts=True)
        n_win = unique_wins.shape[0]
        max_ct = counts.max().item()

        # --- per-window local index (vectorised) -------------------------- #
        order = torch.argsort(inverse)
        sorted_inv = inverse[order]
        offsets = torch.zeros(n_win, dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(0)[:-1]
        positions = torch.arange(N, dtype=torch.long, device=device)
        local_idx = torch.empty(N, dtype=torch.long, device=device)
        local_idx[order] = positions - offsets[sorted_inv]

        # --- pack into (n_win, max_ct, *) --------------------------------- #
        padded = feats.new_zeros(n_win, max_ct, C)
        padded[inverse, local_idx] = feats

        pad_mask = torch.ones(n_win, max_ct, dtype=torch.bool, device=device)
        pad_mask[inverse, local_idx] = False

        packed_xyz = coords.new_zeros(n_win, max_ct, 3)
        packed_xyz[inverse, local_idx] = local_xyz

        # --- QKV & attention ---------------------------------------------- #
        qkv = self.qkv(padded).reshape(n_win, max_ct, 3, H, D)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)    # (W, H, L, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # (W, H, L, L)

        # 3-D relative position bias
        pz, py, px = packed_xyz[..., 0], packed_xyz[..., 1], packed_xyz[..., 2]
        span = 2 * ws - 1
        rel_idx = ((pz.unsqueeze(2) - pz.unsqueeze(1) + ws - 1) * span * span
                   + (py.unsqueeze(2) - py.unsqueeze(1) + ws - 1) * span
                   + (px.unsqueeze(2) - px.unsqueeze(1) + ws - 1))
        bias = self.rel_pos_bias[rel_idx].permute(0, 3, 1, 2)  # (W, H, L, L)
        attn = attn + bias

        attn = attn.masked_fill(pad_mask[:, None, None, :], float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(n_win, max_ct, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # --- scatter back & residual -------------------------------------- #
        out_feats = out[inverse, local_idx]
        out_feats = out_feats + identity

        return x.replace_feature(out_feats)


# --------------------------------------------------------------------------- #
#  SpConvUNet + Local Attention
# --------------------------------------------------------------------------- #
@MODELS.register_module()
class SpConvUNet_Attn(nn.Module):
    """SpConvUNet with window self-attention at selected encoder levels.

    Structurally identical to ``SpConvUNet`` except that a
    ``SparseWindowAttention`` block is appended after the encoder
    residual blocks at every level whose ``indice_key_id`` is listed in
    ``attn_levels``.

    Args:
        num_planes (List[int]): Number of channels at each level.
        norm_fn (Callable): Normalization constructor.
        block_reps (int): Residual blocks per level.
        block (Callable): Block class.
        indice_key_id (int): Id of the current recursion level.
        normalize_before (bool): Whether to use pre-norm.
        return_blocks (bool): Whether to return multi-level features.
        attn_levels (List[int] | None): Which levels (1-indexed,
            matching ``indice_key_id``) should receive attention.
            Defaults to the deepest 3 levels.
        num_heads (int): Attention heads.
        window_size (int): Cubic window side-length in voxels.
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
                 attn_levels=None,
                 num_heads=4,
                 window_size=8):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        if attn_levels is None:
            n_total = len(num_planes) + indice_key_id - 1
            attn_levels = list(range(max(indice_key_id, n_total - 2),
                                     n_total + 1))

        # --- encoder residual blocks ------------------------------------- #
        blocks = OrderedDict({
            f'block{i}': block(
                num_planes[0], num_planes[0], norm_fn,
                normalize_before=normalize_before,
                indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        })
        self.blocks = spconv.SparseSequential(blocks)

        # --- local window attention (optional) ---------------------------- #
        if indice_key_id in attn_levels:
            self.attn = SparseWindowAttention(
                num_planes[0], num_heads=num_heads,
                window_size=window_size)

        # --- deeper levels (recursive) ------------------------------------ #
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

            self.u = SpConvUNet_Attn(
                num_planes[1:], norm_fn, block_reps, block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                attn_levels=attn_levels,
                num_heads=num_heads,
                window_size=window_size)

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

            blocks_tail = OrderedDict({
                f'block{i}': block(
                    num_planes[0] * (2 - i), num_planes[0], norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
                for i in range(block_reps)
            })
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.
            previous_outputs (List[SparseConvTensor]): Previous outputs.

        Returns:
            SparseConvTensor: Output tensor.
        """
        output = self.blocks(input)

        if hasattr(self, 'attn'):
            output = self.attn(output)

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

            output = output.replace_feature(torch.cat(
                (identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output
