"""IAS + BCL + V3: Combined model.

Combines three orthogonal improvements on top of the base
ForAINetV2OneFormer3D_XAwarequery:

1. Instance-Aware Sampling (IAS)
   - Replaces vanilla FPS with cluster-balanced query selection
   - Ensures every potential instance gets query coverage

2. Boundary Contrastive Learning (BCL)
   - Pushes apart embeddings of spatially adjacent cross-boundary voxels
   - Pulls together embeddings of intra-instance boundary voxels

3. V3 Decoder (3D Position Encoding + MLP Mask Head)
   - Sinusoidal PE injected into voxel features so the decoder knows
     WHERE each voxel is in 3D space
   - MLP Mask Head replaces einsum('nd,md->nm') with two learnable
     nonlinear projections

Inheritance:
    ForAINetV2OneFormer3D_XAwarequery
        -> IAS  (_select_query_indices)
            -> IAS_BCL  (loss with boundary contrastive)
                -> IAS_BCL_V3  (extract_feat with voxel coords + V3 decoder)
"""

from mmdet3d.registry import MODELS

from .boundary_contrastive import ForAINetV2OneFormer3D_XAwarequery_IAS_BCL


@MODELS.register_module()
class ForAINetV2OneFormer3D_XAwarequery_IAS_BCL_V3(
        ForAINetV2OneFormer3D_XAwarequery_IAS_BCL):
    """IAS + BCL + V3 model.

    Inherits IAS (instance-aware query sampling) and BCL (boundary
    contrastive loss) from the parent, and adds V3-style voxel
    coordinate tracking + decoder auto-wrapping so that the V3 decoder
    (ForAINetv2QueryDecoder_XAwarequery_V3) receives `coords=` argument
    automatically.

    The only changes are:
    1. extract_feat saves voxel coordinates into self._voxel_coords
    2. decoder is wrapped to auto-inject coords=... into every call

    All loss() / predict() / predict_bm1orbm2() logic is inherited
    unchanged from IAS_BCL.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._voxel_coords = None
        self._raw_decoder = self.decoder
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
