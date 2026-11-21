import logging
from typing import Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stopping hook that monitors per-epoch average training loss
    and stops training when the loss stops decreasing.

    After each training iteration, ``after_train_iter`` collects the ``loss``
    value from the outputs. At the end of each epoch,
    ``after_train_epoch`` computes the epoch-level average loss and compares
    it against the best (lowest) epoch loss seen so far. If the loss does not
    improve for ``patience`` consecutive epochs, training is stopped early.

    Args:
        patience (int): Number of epochs with no improvement
            after which training will be stopped. Defaults to 10.
        min_delta (float): Minimum decrease in loss to qualify as an
            improvement. Defaults to 1e-4.
        min_epoch (int): Minimum number of epochs to run before early
            stopping can be triggered. Defaults to 0.
        save_best (bool): If True, when a new best loss is found the
            current checkpoint is saved to ``work_dir/best.pth``.
            Defaults to True.
        smooth_window (int): Number of recent epochs to average before
            comparing. Setting to > 1 smooths out short-term noise.
            Defaults to 1 (no smoothing).
    """

    priority = 'LOW'

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 min_epoch: int = 0,
                 save_best: bool = True,
                 smooth_window: int = 1) -> None:
        super().__init__()
        self._patience = patience
        self._min_delta = min_delta
        self._min_epoch = min_epoch
        self._save_best = save_best
        self._smooth_window = max(1, int(smooth_window))

        self._best_loss: Optional[float] = None
        self._best_epoch: int = -1
        self._best_ckpt_path: Optional[str] = None
        self._wait_count: int = 0
        self._stopped_epoch: int = 0

        self._epoch_loss_sum: float = 0.0
        self._epoch_loss_count: int = 0

        self._recent_losses: list = []

    def before_train_epoch(self, runner) -> None:
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

    def after_train_iter(self, runner, batch_idx: int,
                         data_batch=None, outputs=None) -> None:
        if outputs is not None and isinstance(outputs, dict):
            loss_val = outputs.get('loss', None)
            if loss_val is not None:
                try:
                    self._epoch_loss_sum += float(loss_val)
                    self._epoch_loss_count += 1
                except (TypeError, ValueError):
                    pass

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        if self._epoch_loss_count == 0:
            print_log(
                '[EarlyStoppingHook] No loss values collected this epoch. '
                'Skipping check.',
                logger='current',
                level=logging.WARNING)
            return

        avg_loss = self._epoch_loss_sum / self._epoch_loss_count
        self._recent_losses.append(avg_loss)

        if len(self._recent_losses) > self._smooth_window:
            self._recent_losses = self._recent_losses[-self._smooth_window:]

        smoothed_loss = sum(self._recent_losses) / len(self._recent_losses)

        print_log(
            f'Average loss: {avg_loss:.4f} (smoothed: {smoothed_loss:.4f})',
            logger='current',
            level=logging.INFO)

        if epoch < self._min_epoch:
            return

        current = smoothed_loss

        if self._best_loss is None:
            self._best_loss = current
            self._best_epoch = epoch
            self._wait_count = 0
            print_log(
                f'Initial best loss: {self._best_loss:.4f} at epoch {epoch}',
                logger='current',
                level=logging.INFO)
            if self._save_best:
                self._save_best_checkpoint(runner)

        elif current < self._best_loss - self._min_delta:
            improvement = self._best_loss - current
            self._best_loss = current
            self._best_epoch = epoch
            self._wait_count = 0
            print_log(
                f'Loss improved ({improvement:+.4f} ↓). '
                f'New best: {current:.4f} at epoch {epoch}.',
                logger='current',
                level=logging.INFO)
            if self._save_best:
                self._save_best_checkpoint(runner)

        else:
            self._wait_count += 1
            print_log(
                f'Loss did not improve. '
                f'Best: {self._best_loss:.4f} (epoch {self._best_epoch}). '
                f'Patience: {self._wait_count}/{self._patience}.',
                logger='current',
                level=logging.INFO)

        if self._wait_count >= self._patience:
            self._stopped_epoch = epoch
            runner.train_loop.stop_training = True
            print_log(
                f'Early stopping triggered at epoch {epoch}. '
                f'Best loss: {self._best_loss:.4f} at epoch {self._best_epoch}.',
                logger='current',
                level=logging.INFO)

    def after_train(self, runner) -> None:
        if self._best_loss is not None:
            print_log(
                f'Training finished. '
                f'Best loss: {self._best_loss:.4f} at epoch {self._best_epoch}.',
                logger='current',
                level=logging.INFO)

    def _save_best_checkpoint(self, runner) -> None:
        import os
        work_dir = getattr(runner, 'work_dir', None)
        if work_dir is None:
            return
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, 'best.pth')
        runner.save_checkpoint(
            save_path,
            by_epoch=True,
            save_optimizer=False,
            save_param_scheduler=False)
        self._best_ckpt_path = save_path
