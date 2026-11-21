"""SpConvUNet with 3D ASPP at the bottleneck layer.

Adds a lightweight Atrous Spatial Pyramid Pooling module at the deepest
level of the recursive U-Net to capture multi-scale context before features
propagate back through the decoder path.
"""
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from torch import nn

from mmdet3d.registry import MODELS
from .spconv_unet import ResidualBlock


class SparseASPP(nn.Module):
    """3D ASPP for sparse convolution tensors.

    Four parallel branches with different receptive fields:
      1. 1x1 SubMConv3d  (point-wise)
      2. 3x3 SubMConv3d  (local context)
      3. Two stacked 3x3 SubMConv3d  (~5x5 receptive field)
      4. Global average pooling + linear + broadcast  (global context)

    Outputs are concatenated and projected back to the input channel count.
    A residual connection wraps the entire block for training stability.

    Args:
        channels (int): Number of input / output channels.
        mid_channels (int or None): Per-branch hidden width.
            Defaults to ``channels // 4``.
        indice_key_prefix (str): Prefix for spconv indice keys to avoid
            collisions with the rest of the network.
    """

    def __init__(self, channels, mid_channels=None,
                 indice_key_prefix='aspp'):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels // 4

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.branch1 = spconv.SparseSequential(
            spconv.SubMConv3d(channels, mid_channels, kernel_size=1,
                              bias=False, indice_key=f'{indice_key_prefix}_1'),
            norm_fn(mid_channels), nn.ReLU())

        self.branch2 = spconv.SparseSequential(
            spconv.SubMConv3d(channels, mid_channels, kernel_size=3,
                              padding=1, bias=False,
                              indice_key=f'{indice_key_prefix}_3'),
            norm_fn(mid_channels), nn.ReLU())

        self.branch3 = spconv.SparseSequential(
            spconv.SubMConv3d(channels, mid_channels, kernel_size=3,
                              padding=1, bias=False,
                              indice_key=f'{indice_key_prefix}_5a'),
            norm_fn(mid_channels), nn.ReLU(),
            spconv.SubMConv3d(mid_channels, mid_channels, kernel_size=3,
                              padding=1, bias=False,
                              indice_key=f'{indice_key_prefix}_5b'),
            norm_fn(mid_channels), nn.ReLU())

        self.global_branch = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            norm_fn(mid_channels), nn.ReLU())

        self.proj = spconv.SparseSequential(
            spconv.SubMConv3d(mid_channels * 4, channels, kernel_size=1,
                              bias=False,
                              indice_key=f'{indice_key_prefix}_proj'),
            norm_fn(channels), nn.ReLU())

    def forward(self, x):
        """Forward pass.

        Args:
            x (SparseConvTensor): Input sparse tensor.

        Returns:
            SparseConvTensor: Output with same indices / spatial_shape.
        """
        identity = x.features

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # --- global context branch ---
        feats = x.features                          # (N, C)
        batch_ids = x.indices[:, 0]                 # (N,)
        uniq_batches = batch_ids.unique()

        # Pool per-batch, stack into (B, C) so BatchNorm sees B > 1
        pooled_list = []
        batch_order = []
        for bid in uniq_batches:
            mask = batch_ids == bid
            pooled_list.append(feats[mask].mean(dim=0))       # (C,)
            batch_order.append(bid)
        pooled_all = torch.stack(pooled_list, dim=0)          # (B, C)
        if pooled_all.size(0) == 1 and self.training:
            # BatchNorm1d needs B > 1; temporarily switch to eval mode
            self.global_branch.eval()
            projected_all = self.global_branch(pooled_all)    # (1, mid)
            self.global_branch.train()
        else:
            projected_all = self.global_branch(pooled_all)    # (B, mid)

        # Broadcast each batch's global feature back to its points
        bid_to_idx = {bid.item(): i for i, bid in enumerate(batch_order)}
        idx_per_point = torch.tensor(
            [bid_to_idx[b.item()] for b in batch_ids],
            device=feats.device, dtype=torch.long)
        global_feats = projected_all[idx_per_point]           # (N, mid)

        cat_feats = torch.cat(
            [b1.features, b2.features, b3.features, global_feats], dim=1)

        fused = x.replace_feature(cat_feats)
        fused = self.proj(fused)

        out = fused.replace_feature(fused.features + identity)
        return out


@MODELS.register_module()
class SpConvUNet_ASPP(nn.Module):
    """SpConv U-Net with 3D ASPP at the bottleneck.

    Identical to ``SpConvUNet`` except that a ``SparseASPP`` block is applied
    after the residual blocks at the deepest recursion level
    (``len(num_planes) == 1``).

    Args:
        num_planes (List[int]): Number of channels in each level.
        norm_fn (Callable): Normalization function constructor.
        block_reps (int): Times to repeat each block.
        block (Callable): Block base class.
        indice_key_id (int): Id of current level.
        normalize_before (bool): Whether to call norm before conv.
        return_blocks (bool): Whether to return previous blocks.
        aspp_mid_channels (int or None): Per-branch width inside ASPP.
            Defaults to ``bottleneck_channels // 4``.
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
                 aspp_mid_channels=None):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        blocks = {
            f'block{i}': block(
                num_planes[0],
                num_planes[0],
                norm_fn,
                normalize_before=normalize_before,
                indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(num_planes) == 1:
            self.aspp = SparseASPP(
                num_planes[0], mid_channels=aspp_mid_channels,
                indice_key_prefix=f'aspp{indice_key_id}')

        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[1]), nn.ReLU())

            self.u = SpConvUNet_ASPP(
                num_planes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                aspp_mid_channels=aspp_mid_channels)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    num_planes[0] * (2 - i),
                    num_planes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.
            previous_outputs (List[SparseConvTensor]): Previous input tensors.

        Returns:
            SparseConvTensor: Output tensor.
        """
        output = self.blocks(input)

        if len(self.num_planes) == 1:
            output = self.aspp(output)

        identity = spconv.SparseConvTensor(output.features, output.indices,
                                           output.spatial_shape,
                                           output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output
