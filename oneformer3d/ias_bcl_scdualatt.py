"""IAS + BCL + SCDualAtt: Combined model.

Combines three orthogonal improvements on top of the base
ForAINetV2OneFormer3D_XAwarequery:

1. Instance-Aware Sampling (IAS)
   - Replaces vanilla FPS with cluster-balanced query selection
   - Ensures every potential instance gets query coverage

2. Boundary Contrastive Learning (BCL)
   - Pushes apart embeddings of spatially adjacent cross-boundary voxels
   - Pulls together embeddings of intra-instance boundary voxels

3. Spatial-Channel Dual Attention (SCDualAtt)
   - Channel branch: GAP -> MLP -> Sigmoid  (global context)
   - Spatial branch: Sparse 1x1 -> 3x3 Conv -> Sigmoid  (local context)
   - Inserted after U-Net backbone, before output BN+ReLU

Inheritance:
    ForAINetV2OneFormer3D_XAwarequery
        -> IAS  (_select_query_indices)
            -> IAS_BCL  (loss with boundary contrastive)
                -> IAS_BCL_SCDualAtt  (extract_feat with dual attention)
"""

from mmdet3d.registry import MODELS

from .boundary_contrastive import ForAINetV2OneFormer3D_XAwarequery_IAS_BCL
from .sc_dual_attention import SCDualAtt


@MODELS.register_module()
class ForAINetV2OneFormer3D_IAS_BCL_SCDualAtt(
        ForAINetV2OneFormer3D_XAwarequery_IAS_BCL):
    """IAS + BCL + Spatial-Channel Dual Attention.

    Inherits IAS (instance-aware query sampling) and BCL (boundary
    contrastive loss) from the parent, and adds SCDualAtt to the
    feature extraction pipeline.

    Extra config keys (on top of IAS_BCL):
        sc_dual_att (dict | None): Config for SCDualAtt module.
            Example: dict(type='SCDualAtt', in_channels=32, reduction=4).
            If None, uses default SCDualAtt(in_channels=num_channels).
    """

    def __init__(self, sc_dual_att=None, **kwargs):
        super().__init__(**kwargs)
        if sc_dual_att is not None:
            sc_dual_att_cfg = dict(sc_dual_att)
            sc_dual_att_cfg.setdefault('in_channels', kwargs['num_channels'])
            self.sc_dual_att = MODELS.build(sc_dual_att_cfg)
        else:
            self.sc_dual_att = SCDualAtt(in_channels=kwargs['num_channels'])

    def extract_feat(self, x):
        """Extract features with SCDualAtt inserted after U-Net."""
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.sc_dual_att(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out
