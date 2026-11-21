#!/usr/bin/env python3
"""Check ForAINetV2 generated dataset (pkl + points/semantic/instance masks).

Supports:
  - mmengine pkl: legacy list[dict] or dict with data_list + metainfo (v2)
  - Points from data_root/points/*.bin (float32, N x 3) or forainetv2_instance_data/*_vert.npy

Example:
  python tools/check_forainetv2_dataset.py --data-root data/ForAINetV2 \\
      --pkl data/ForAINetV2/forainetv2_oneformer3d_infos_train.pkl --max-samples 20
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import mmengine
except ImportError as e:
    raise SystemExit(
        'Please install mmengine (e.g. pip install mmengine).') from e

# Matches configs / ForAINetV2SegDataset_ METAINFO: ground, wood, leaf
VALID_SEMANTIC = {0, 1, 2}


def _load_pkl(path: Path):
    raw = mmengine.load(str(path))
    if isinstance(raw, dict) and 'data_list' in raw:
        return raw['data_list'], raw.get('metainfo')
    return raw, None


def _get_paths(info: dict):
    """Normalize paths from legacy or v2 info dict."""
    if 'lidar_points' in info and info['lidar_points'].get('lidar_path'):
        lp = info['lidar_points']
        pts_name = lp['lidar_path']
        nfeat = lp.get('num_pts_feats', 3)
    else:
        pts_name = info.get('pts_path', '')
        pc = info.get('point_cloud', {})
        nfeat = pc.get('num_features', 3)

    sem_name = info.get('pts_semantic_mask_path')
    ins_name = info.get('pts_instance_mask_path')

    if isinstance(sem_name, Path):
        sem_name = sem_name.name
    if isinstance(ins_name, Path):
        ins_name = ins_name.name
    if isinstance(pts_name, Path):
        pts_name = pts_name.name

    return str(pts_name), str(sem_name) if sem_name else None, str(
        ins_name) if ins_name else None, int(nfeat)


def _load_points(
    data_root: Path,
    pts_name: str,
    nfeat: int,
    npy_dir: Path,
    points_subdir: str = 'points',
) -> np.ndarray:
    rel = pts_name.replace('\\', '/')
    fname = Path(rel).name
    for bin_path in (data_root / rel, data_root / points_subdir / fname):
        if bin_path.is_file():
            raw = np.fromfile(bin_path, dtype=np.float32)
            if raw.size % nfeat != 0:
                raise ValueError(
                    f'{bin_path}: size {raw.size} not divisible by '
                    f'nfeat={nfeat}')
            return raw.reshape(-1, nfeat)

    stem = Path(pts_name).stem
    npy_path = npy_dir / f'{stem}_vert.npy'
    if npy_path.is_file():
        pts = np.load(npy_path)
        if pts.ndim != 2 or pts.shape[1] < nfeat:
            raise ValueError(f'{npy_path}: bad shape {pts.shape}')
        return pts[:, :nfeat].astype(np.float32)

    raise FileNotFoundError(
        f'Missing points: tried {data_root / rel}, '
        f'{data_root / points_subdir / fname}, {npy_path}')


def _load_mask(
    data_root: Path,
    name: str | None,
    npy_dir: Path,
    stem: str,
    suffix: str,
    subdir: str,
) -> np.ndarray:
    if not name:
        raise ValueError('mask path missing in info')
    rel = name.replace('\\', '/')
    fname = Path(rel).name
    candidates = [
        data_root / rel,
        data_root / subdir / fname,
    ]
    for bin_path in candidates:
        if bin_path.is_file():
            return np.fromfile(bin_path, dtype=np.int64)

    npy_path = npy_dir / f'{stem}_{suffix}.npy'
    if npy_path.is_file():
        return np.load(npy_path).astype(np.int64)

    tried = ', '.join(str(p) for p in candidates)
    raise FileNotFoundError(
        f'Missing mask ({suffix}): tried {tried} and {npy_path}')


def _check_sample(
    data_root: Path,
    info: dict,
    idx: int,
    npy_dir: Path,
    semantic_subdir: str,
    instance_subdir: str,
    points_subdir: str,
) -> tuple[list[str], dict]:
    issues: list[str] = []
    stats: dict = {}

    pts_name, sem_name, ins_name, nfeat = _get_paths(info)
    stem = Path(pts_name).stem

    try:
        pts = _load_points(data_root, pts_name, nfeat, npy_dir,
                             points_subdir)
    except Exception as e:
        issues.append(f'points load failed: {e}')
        return issues, stats

    try:
        sem = _load_mask(data_root, sem_name, npy_dir, stem, 'sem_label',
                           semantic_subdir)
        ins = _load_mask(data_root, ins_name, npy_dir, stem, 'ins_label',
                           instance_subdir)
    except Exception as e:
        issues.append(f'mask load failed: {e}')
        return issues, stats

    n = pts.shape[0]
    stats['num_points'] = int(n)

    if sem.shape[0] != n:
        issues.append(
            f'semantic length {sem.shape[0]} != num_points {n}')
    if ins.shape[0] != n:
        issues.append(
            f'instance length {ins.shape[0]} != num_points {n}')

    bad_sem = np.setdiff1d(np.unique(sem), list(VALID_SEMANTIC))
    if bad_sem.size > 0:
        issues.append(f'unexpected semantic ids: {bad_sem.tolist()}')

    u_sem = Counter(sem.tolist())
    stats['semantic_hist'] = dict(u_sem)

    inst_pos = ins > 0
    inst_ids = set(np.unique(ins[inst_pos]).tolist()) if inst_pos.any() else set()
    stats['num_distinct_instances'] = len(inst_ids)
    stats['points_with_instance'] = int(inst_pos.sum())

    # Ground (0) should not carry instance id > 0
    ground = sem == 0
    if np.any(ground & (ins > 0)):
        bad = int((ground & (ins > 0)).sum())
        issues.append(
            f'ground semantic but instance>0 on {bad} points')

    # Thing classes: wood/leaf usually have instance > 0 (allow some 0 if sparse noise)
    thing = (sem == 1) | (sem == 2)
    if thing.any():
        thing_no_inst = thing & (ins == 0)
        ratio = float(thing_no_inst.sum()) / float(thing.sum())
        stats['thing_points_without_inst_ratio'] = ratio
        if ratio > 0.5:
            issues.append(
                f'>{50}% of wood/leaf points have instance id 0 '
                f'(ratio={ratio:.3f}); check labels')

    return issues, stats


def main():
    parser = argparse.ArgumentParser(
        description='Validate ForAINetV2 pkl and on-disk arrays.')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/ForAINetV2',
        help='Dataset root (contains points/, semantic_mask/, pkl, etc.)')
    parser.add_argument(
        '--pkl',
        type=str,
        required=True,
        help='Path to forainetv2_oneformer3d_infos_*.pkl')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Max scenes to check (0 = all)')
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start index in data_list')
    parser.add_argument(
        '--semantic-subdir',
        type=str,
        default='semantic_mask',
        help='Same as data_prefix pts_semantic_mask in config')
    parser.add_argument(
        '--instance-subdir',
        type=str,
        default='instance_mask',
        help='Same as data_prefix pts_instance_mask in config')
    parser.add_argument(
        '--points-subdir',
        type=str,
        default='points',
        help='Same as data_prefix pts in config')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    pkl_path = Path(args.pkl)
    if not pkl_path.is_file():
        pkl_path = (data_root / args.pkl).resolve()
    if not pkl_path.is_file():
        raise SystemExit(f'PKL not found: {args.pkl}')

    npy_dir = data_root / 'forainetv2_instance_data'

    data_list, metainfo = _load_pkl(pkl_path)
    n_total = len(data_list)
    end = n_total if args.max_samples <= 0 else min(
        n_total, args.start + args.max_samples)

    print(f'PKL: {pkl_path}')
    print(f'scenes in file: {n_total}; checking [{args.start}, {end})')
    if metainfo:
        print(f'metainfo keys: {list(metainfo.keys())}')

    all_issues = 0
    sem_global = Counter()
    inst_counts = []

    for i in range(args.start, end):
        info = data_list[i]
        issues, stats = _check_sample(
            data_root, info, i, npy_dir,
            args.semantic_subdir, args.instance_subdir, args.points_subdir)
        pts_name, _, _, _ = _get_paths(info)
        tag = Path(pts_name).stem

        if issues:
            all_issues += len(issues)
            print(f'[{i}] {tag}  FAIL')
            for m in issues:
                print(f'    - {m}')
        else:
            print(f'[{i}] {tag}  OK  points={stats.get("num_points")}  '
                  f'instances={stats.get("num_distinct_instances")}')

        if 'semantic_hist' in stats:
            sem_global.update(stats['semantic_hist'])
        if 'num_distinct_instances' in stats:
            inst_counts.append(stats['num_distinct_instances'])

    print('--- summary ---')
    print(f'checked: {end - args.start} scenes, issue lines: {all_issues}')
    if sem_global:
        print('semantic count (sum over checked scenes):',
              dict(sorted(sem_global.items())))
    if inst_counts:
        print(
            'instances per scene: min', min(inst_counts), 'max',
            max(inst_counts), 'mean', float(np.mean(inst_counts)))

    if all_issues > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
