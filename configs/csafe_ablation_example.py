# =============================================================================
#  Example config for CSAFE fusion-mechanism ablation experiments
# =============================================================================
# Usage: change ``csafe_type`` to one of
#   'sum'       →  (a) Simple sum
#   'concat'    →  (b) Concat + linear projection
#   'scalar'    →  (c) Learnable scalar weights
#   'no_query'  →  (d) No query projection (linear alignment only)
#   'no_key'    →  (e) No key projections (raw branch features as keys)
#   'full'      →  (f) Full CSAFE
# =============================================================================

# ––– Run all six in parallel or sequentially by changing csafe_type –––
_base_ = './train_custom_forest.py'

num_channels = 32

model = dict(
    backbone=dict(
        type='SpConvUNet_CSAFE_Ablation',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
        csafe_k=16,
        csafe_mid_channels=32,
        csafe_levels=[2, 3],
        csafe_type='full',   # <--- change this to 'sum'/'concat'/'scalar'/etc.
    ),
)
