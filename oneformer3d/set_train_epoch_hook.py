from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SetTrainEpochHook(Hook):
    """Sync ``runner.epoch`` onto models that implement ``set_train_epoch``."""

    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'set_train_epoch'):
            model.set_train_epoch(int(runner.epoch))
