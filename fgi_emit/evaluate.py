"""Run the unchanged, pinned official evaluator in an isolated process."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from .protocol import SPLITS

# FGI-EMIT's trusted, checksum-pinned plot_data.yaml contains metadata for
# 1,561 trees and exceeds OmegaConf's default 10,000-node safety limit.
# Keep a finite upper bound rather than disabling the protection entirely.
os.environ.setdefault('OMEGACONF_MAX_YAML_EXPANDED_NODES', '100000')


def _install_numpy_compat():
    """Provide aliases removed in NumPy 1.24 for the pinned evaluator."""
    # Access np.__dict__ directly so checking the missing alias does not emit
    # NumPy's FutureWarning from ``hasattr(np, 'bool')``.
    if 'bool' not in np.__dict__:
        np.bool = np.bool_


def _to_builtin(value):
    """Convert NumPy scalars before the official code builds DictConfig."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def evaluate(pred_root, raw_root, split, output, official_dir=None):
    from omegaconf import OmegaConf

    _install_numpy_compat()
    pred_root, raw_root = Path(pred_root).resolve(), Path(raw_root).resolve()
    official_dir = Path(official_dir or raw_root / 'accuracy').resolve()
    if not (official_dir / 'accuracy.py').exists():
        raise FileNotFoundError('Official evaluator missing; run python -m fgi_emit.download')
    expected = {f'plot_{i}.las' for i in SPLITS[split]}
    actual = {p.name for p in (pred_root / split).glob('*.las')}
    if actual != expected:
        raise ValueError(f'Refusing incomplete/stale {split} evaluation: '
                         f'missing={expected-actual}, extra={actual-expected}')
    sys.path.insert(0, str(official_dir))
    import accuracy as official_accuracy

    # OmegaConf rejects numpy.float64 in the official metric_dict. Wrap only
    # the constructor reference used by accuracy.py; metric computation stays
    # byte-for-byte identical to the released evaluator.
    official_dict_config = official_accuracy.DictConfig
    official_accuracy.DictConfig = lambda value: official_dict_config(
        _to_builtin(value))
    compute_accuracy_metrics = official_accuracy.compute_accuracy_metrics

    cfg = OmegaConf.load(official_dir / 'cfgs' / 'accuracy.yaml')
    cfg.data_path = str(pred_root)
    cfg.plot_metadata_path = str(raw_root)
    cfg.id_list = list(SPLITS[split])
    # Keep the released defaults: IoU=0.5, small-instance filter 40 points/1.5 m,
    # and all seven metric groups, including AP50 and crown categories.
    result = compute_accuracy_metrics(cfg, custom_split_name=split,
                                      verbose=True, return_metrics=True)
    if result is None:
        raise RuntimeError('Official evaluator produced no metrics.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(OmegaConf.to_container(result), indent=2), encoding='utf-8')
    OmegaConf.save(cfg, str(output.with_suffix('.protocol.yaml')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, default=Path('data/FGI_EMIT/raw'))
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-dir', type=Path)
    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == '__main__':
    main()
