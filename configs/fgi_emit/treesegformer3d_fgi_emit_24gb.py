"""FGI-EMIT TreeSegFormer3D settings for a 24 GB GPU.

The paper-scale 12 m crop is retained. A physical batch of one with two-step
gradient accumulation approximates the effective batch size of two while
avoiding the memory peak of two simultaneous samples. MMEngine advances the
iteration scheduler per dataloader iteration, hence 6,500 x 10 = 65,000.
"""

_base_ = ['./treesegformer3d_fgi_emit.py']

work_dir = 'work_dirs/fgi_emit/treesegformer3d_paper_24gb'

train_dataloader = dict(batch_size=1)
optim_wrapper = dict(accumulative_counts=2)
param_scheduler = dict(end=65000)

val_evaluator = dict(
    pred_root='work_dirs/fgi_emit/treesegformer3d_paper_24gb/predictions')
test_evaluator = dict(
    pred_root='work_dirs/fgi_emit/treesegformer3d_paper_24gb/predictions')
