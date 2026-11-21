# =============================================================================
# SpConvUNet backbone with configurable CSAFE variant.
# Switch the fusion mechanism via the config key ``csafe_type``.
# =============================================================================
import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
import torch.nn as nn

from mmdet3d.registry import MODELS
from .spconv_unet import ResidualBlock


@MODELS.register_module()
class SpConvUNet_CSAFE_Ablation(nn.Module):
    """SpConvUNet backbone with selectable CSAFE variant at encoder levels.

    Identical to ``SpConvUNet_GFEM`` except the fusion variant is
    controlled by the ``csafe_type`` config key.

    Args:
        num_planes (List[int]): Channels per level.
        norm_fn: Normalisation constructor.
        block_reps (int): Residual block repetitions.
        block: Block class.
        indice_key_id (int): SpConv indice key counter.
        normalize_before (bool): BN-ReLU before conv.
        return_blocks (bool): Return intermediate block outputs.
        csafe_k (int): KNN neighbours.
        csafe_mid_channels (int): Attention dimension.
        csafe_levels (list[int] | int | None): Which levels to enable CSAFE.
        csafe_type (str): One of:
            ``'sum'``          →  CSAFE_SumFusion          (a)  original GFEM, direct sum
            ``'concat'``       →  CSAFE_ConcatProj          (b)  concat + linear projection
            ``'full'``         →  CSAFE                     (c)  full CSAFE adaptive fusion
    """

    _CSAFE_REGISTRY = {
        'sum':       'CSAFE_SumFusion',
        'concat':    'CSAFE_ConcatProj',
        'full':      'CSAFE',
    }

    def __init__(self,
                 num_planes,
                 norm_fn=functools.partial(
                     nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2,
                 block=ResidualBlock,
                 indice_key_id=1,
                 normalize_before=True,
                 return_blocks=False,
                 csafe_k=16,
                 csafe_mid_channels=32,
                 csafe_levels=None,
                 csafe_type='full'):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        if isinstance(block, str):
            assert block in ('residual', 'vgg', 'asym')
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

        # ---------- CSAFE variant ----------
        if csafe_levels is not None:
            self._csafe_level_set = set(csafe_levels)
        else:
            self._csafe_level_set = None  # all

        self.use_csafe = (self._csafe_level_set is None
                          or indice_key_id in self._csafe_level_set)

        if self.use_csafe:
            csafe_cls_name = self._CSAFE_REGISTRY.get(csafe_type, 'CSAFE')
            csafe_cls = MODELS.get(csafe_cls_name)
            self.csafe = csafe_cls(
                num_planes[0], mid_channels=csafe_mid_channels, k=csafe_k)

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

            self.u = SpConvUNet_CSAFE_Ablation(
                num_planes[1:], norm_fn, block_reps, block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
                csafe_k=csafe_k,
                csafe_mid_channels=csafe_mid_channels,
                csafe_levels=csafe_levels,
                csafe_type=csafe_type)

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
        output = self.blocks(input)

        if self.use_csafe:
            output = self.csafe(output)

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
