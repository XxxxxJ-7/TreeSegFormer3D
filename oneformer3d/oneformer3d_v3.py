"""
V3 Model class — minimal override.

Only two things change from baseline:
1. extract_feat saves voxel coordinates into self._voxel_coords
2. decoder is wrapped to auto-inject coords=... into every call

All loss() / predict() / predict_bm1orbm2() logic is inherited unchanged.
"""

import spconv.pytorch as spconv
from mmdet3d.registry import MODELS

from .oneformer3d import ForAINetV2OneFormer3D_XAwarequery


@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_V3(
    ForAINetV2OneFormer3D_XAwarequery):
    """V3 model that passes 3D voxel coordinates to the V3 decoder."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._voxel_coords = None
        self._raw_decoder = self.decoder
        # Use object.__setattr__ to bypass nn.Module's type check
        object.__setattr__(self, 'decoder', self._decoder_with_coords)

    def _decoder_with_coords(self, *args, **kwargs):
        """Auto-inject voxel coordinates into every decoder call."""
        return self._raw_decoder(*args, coords=self._voxel_coords, **kwargs)

    def extract_feat(self, x):
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        coords = []
        for i in x.indices[:, 0].unique():
            mask = x.indices[:, 0] == i
            out.append(x.features[mask])
            coords.append(x.indices[mask][:, 1:].float())
        self._voxel_coords = coords
        return out
