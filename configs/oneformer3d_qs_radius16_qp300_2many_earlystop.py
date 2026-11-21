# This config extends the base config with early stopping.
#
# Why monitor training loss instead of validation metrics?
#   During training, validation predict() does NOT use FPS for query
#   sampling (it just takes the first N features), unlike training
#   which uses FPS.  Moreover validation data are small CylinderCrop
#   patches (radius=6m).  Therefore val AP/mIoU do NOT reflect
#   real model quality — only training loss is reliable.
#
# EarlyStoppingHook:
#   - Collects 'loss' from after_train_iter outputs every iteration.
#   - At end of epoch, computes epoch-average loss.
#   - Applies smooth_window smoothing before comparison.
#   - Stops when loss hasn't improved for patience consecutive epochs.

_base_ = [
    './oneformer3d_qs_radius16_qp300_2many.py',
]

custom_hooks = [
    dict(type='SetTrainEpochHook'),
    dict(type='EmptyCacheHook', after_iter=True),
    dict(
        type='EarlyStoppingHook',
        patience=30,          # Stop after N epochs with no improvement
        min_delta=1e-4,       # Minimum loss decrease to count as improvement
        min_epoch=200,        # Don't trigger before this epoch
        save_best=True,       # Save best checkpoint to work_dir/best.pth
        smooth_window=3),     # Average over last 3 epochs to reduce noise
]
