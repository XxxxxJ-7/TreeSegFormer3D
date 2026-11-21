"""Prepare LAS/LAZ forest point cloud data for ForestFormer3D training.

Semantic classes:
    0: ground  (stuff, treeID == 0)
    1: tree    (thing, treeID != 0, each unique treeID is one instance)

Usage 1 - Explicitly specify files for each split:
    python tools/prepare_las_data.py \
        --input-dir /path/to/las/files \
        --output-dir data/CustomForest \
        --train-files plot1,plot2,plot3 \
        --val-files plot4 \
        --test-files plot5

Usage 2 - Random split by ratio:
    python tools/prepare_las_data.py \
        --input-dir /path/to/las/files \
        --output-dir data/CustomForest \
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

File names can be specified with or without the .las extension.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import mmengine
import numpy as np


def read_las_file(las_path, tree_id_field='treeID'):
    """Read a LAS/LAZ file and extract xyz coordinates and treeID.

    Args:
        las_path (str): Path to the LAS/LAZ file.
        tree_id_field (str): Name of the treeID field in the LAS file.

    Returns:
        tuple: (xyz, tree_id) where xyz is (N, 3) float32 and
               tree_id is (N,) int64.
    """
    try:
        import laspy
    except ImportError:
        print('Please install laspy:  pip install laspy lazrs')
        sys.exit(1)

    las = laspy.read(las_path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)

    field_candidates = [tree_id_field, 'treeID', 'TreeID', 'tree_id',
                        'treeId', 'TreeId', 'treeid']
    tree_id = None
    for name in field_candidates:
        try:
            tree_id = np.array(getattr(las, name), dtype=np.int64)
            break
        except AttributeError:
            continue

    if tree_id is None:
        dims = list(las.point_format.dimension_names)
        raise ValueError(
            f'Cannot find tree ID field in {las_path}. '
            f'Tried: {field_candidates}. '
            f'Available dimensions: {dims}')

    return xyz, tree_id


def process_single_las(las_path, output_dir, tree_id_field, center_xy=True):
    """Process one LAS file into the ForestFormer3D binary format.

    Args:
        las_path (str): Path to the LAS file.
        output_dir (str): Root output directory.
        tree_id_field (str): Name of the treeID attribute.
        center_xy (bool): Whether to center XY to the scene mean.

    Returns:
        dict: Sample info dict, or None if the file has no valid points.
    """
    sample_id = Path(las_path).stem
    xyz, tree_id = read_las_file(las_path, tree_id_field)

    if len(xyz) == 0:
        print(f'  [SKIP] {sample_id}: empty point cloud')
        return None

    if center_xy:
        xyz[:, :2] -= xyz[:, :2].mean(axis=0)
    xyz = xyz.astype(np.float32)

    # --- semantic mask: 0 = ground, 1 = tree ---
    semantic_mask = np.where(tree_id != 0, 1, 0).astype(np.int64)

    # --- instance mask: remap treeIDs to consecutive ints ---
    instance_mask = np.zeros(len(tree_id), dtype=np.int64)
    tree_points = semantic_mask == 1
    if tree_points.any():
        unique_tree_ids = np.unique(tree_id[tree_points])
        id_map = {old: new + 1 for new, old in enumerate(unique_tree_ids)}
        instance_mask[tree_points] = np.array(
            [id_map[tid] for tid in tree_id[tree_points]], dtype=np.int64)

    n_trees = len(np.unique(tree_id[tree_points])) if tree_points.any() else 0
    n_ground = int((semantic_mask == 0).sum())
    n_tree_pts = int(tree_points.sum())

    # --- save binary files ---
    xyz.tofile(os.path.join(output_dir, 'points', f'{sample_id}.bin'))
    semantic_mask.tofile(
        os.path.join(output_dir, 'semantic_mask', f'{sample_id}.bin'))
    instance_mask.tofile(
        os.path.join(output_dir, 'instance_mask', f'{sample_id}.bin'))

    print(f'  {sample_id}: {len(xyz)} pts | '
          f'{n_trees} trees | '
          f'ground {n_ground} | tree {n_tree_pts}')

    return dict(
        sample_id=sample_id,
        num_points=len(xyz),
        num_trees=n_trees,
    )


def parse_file_list(raw_str):
    """Parse a comma-separated list of file names (with or without .las)."""
    names = [n.strip() for n in raw_str.split(',') if n.strip()]
    return [Path(n).stem for n in names]


def resolve_las_path(input_dir, stem):
    """Find the actual LAS/LAZ path for a given stem."""
    for ext in ('.las', '.LAS', '.laz', '.LAZ'):
        p = os.path.join(input_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def split_samples_random(sample_ids, train_ratio, val_ratio, seed=42):
    """Split sample IDs into train / val / test by ratio."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(sample_ids))

    n_train = max(1, int(len(indices) * train_ratio))
    n_val = max(1, int(len(indices) * val_ratio))

    train = [sample_ids[i] for i in indices[:n_train]]
    val = [sample_ids[i] for i in indices[n_train:n_train + n_val]]
    test = [sample_ids[i] for i in indices[n_train + n_val:]]
    if len(test) == 0:
        test = val[:1]

    return dict(train=train, val=val, test=test)


def create_info_pkl(output_dir, split_name, sample_ids):
    """Create an MMEngine-v2 info pkl for one split."""
    data_list = []
    for sid in sample_ids:
        data_info = dict(
            lidar_points=dict(num_pts_feats=3, lidar_path=f'{sid}.bin'),
            pts_semantic_mask_path=f'{sid}.bin',
            pts_instance_mask_path=f'{sid}.bin',
            sample_idx=sid,
            instances=[],
        )
        data_list.append(data_info)

    metainfo = dict(
        categories={'tree': 0},
        dataset='custom_forest',
        info_version='1.1',
    )
    info = dict(metainfo=metainfo, data_list=data_list)
    pkl_path = os.path.join(
        output_dir, f'forainetv2_oneformer3d_infos_{split_name}.pkl')
    mmengine.dump(info, pkl_path, 'pkl')
    print(f'  {split_name} pkl saved -> {pkl_path}  ({len(sample_ids)} scenes)')


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAS forest data to ForestFormer3D format')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing .las / .laz files')
    parser.add_argument('--output-dir', default='data/CustomForest',
                        help='Output root (default: data/CustomForest)')
    parser.add_argument('--tree-id-field', default='treeID',
                        help='Name of the treeID attribute in LAS files')

    # --- Mode 1: explicitly specify files per split ---
    split_group = parser.add_argument_group('Explicit split (mode 1)')
    split_group.add_argument(
        '--train-files', type=str, default=None,
        help='Comma-separated file names for training, e.g. "plot1,plot2"')
    split_group.add_argument(
        '--val-files', type=str, default=None,
        help='Comma-separated file names for validation, e.g. "plot3"')
    split_group.add_argument(
        '--test-files', type=str, default=None,
        help='Comma-separated file names for testing, e.g. "plot4"')

    # --- Mode 2: random split by ratio ---
    ratio_group = parser.add_argument_group('Random split (mode 2)')
    ratio_group.add_argument('--train-ratio', type=float, default=0.7)
    ratio_group.add_argument('--val-ratio', type=float, default=0.15)
    ratio_group.add_argument('--test-ratio', type=float, default=0.15)
    ratio_group.add_argument('--seed', type=int, default=42)

    parser.add_argument('--no-center', action='store_true',
                        help='Do not center XY coordinates')
    args = parser.parse_args()

    explicit_mode = any(x is not None for x in
                        [args.train_files, args.val_files, args.test_files])

    if explicit_mode:
        if args.train_files is None:
            parser.error('--train-files is required when using explicit mode')
        if args.val_files is None:
            parser.error('--val-files is required when using explicit mode')
        if args.test_files is None:
            parser.error('--test-files is required when using explicit mode')
    else:
        assert abs(args.train_ratio + args.val_ratio
                   + args.test_ratio - 1.0) < 1e-6, \
            'train/val/test ratios must sum to 1.0'

    # --- build the list of files to process ---
    if explicit_mode:
        split_stems = dict(
            train=parse_file_list(args.train_files),
            val=parse_file_list(args.val_files),
            test=parse_file_list(args.test_files),
        )
        all_stems = []
        for v in split_stems.values():
            all_stems.extend(v)
        if len(all_stems) != len(set(all_stems)):
            parser.error('Duplicate file names across splits')

        las_files = []
        for stem in all_stems:
            p = resolve_las_path(args.input_dir, stem)
            if p is None:
                print(f'ERROR: cannot find LAS/LAZ file for "{stem}" '
                      f'in {args.input_dir}')
                sys.exit(1)
            las_files.append(p)
    else:
        las_files = sorted(
            glob.glob(os.path.join(args.input_dir, '*.las'))
            + glob.glob(os.path.join(args.input_dir, '*.LAS'))
            + glob.glob(os.path.join(args.input_dir, '*.laz'))
            + glob.glob(os.path.join(args.input_dir, '*.LAZ')))

    if not las_files:
        print(f'ERROR: no .las/.laz files found in {args.input_dir}')
        sys.exit(1)
    print(f'Found {len(las_files)} LAS/LAZ files to process\n')

    # --- create output directories ---
    for sub in ('points', 'instance_mask', 'semantic_mask', 'meta_data'):
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # --- process each file ---
    sample_ids = []
    total_points = 0
    total_trees = 0
    for las_path in las_files:
        info = process_single_las(
            las_path, args.output_dir, args.tree_id_field,
            center_xy=not args.no_center)
        if info is not None:
            sample_ids.append(info['sample_id'])
            total_points += info['num_points']
            total_trees += info['num_trees']

    if not sample_ids:
        print('ERROR: no valid scenes produced')
        sys.exit(1)

    print(f'\nTotal: {len(sample_ids)} scenes, '
          f'{total_points:,} points, {total_trees:,} tree instances\n')

    # --- train / val / test split ---
    if explicit_mode:
        processed = set(sample_ids)
        splits = {}
        for split_name, stems in split_stems.items():
            valid = [s for s in stems if s in processed]
            skipped = [s for s in stems if s not in processed]
            if skipped:
                print(f'  WARNING: skipped from {split_name}: {skipped}')
            splits[split_name] = valid
    else:
        splits = split_samples_random(
            sample_ids, args.train_ratio, args.val_ratio, seed=args.seed)

    for split_name, ids in splits.items():
        list_path = os.path.join(
            args.output_dir, 'meta_data', f'{split_name}_list.txt')
        with open(list_path, 'w') as f:
            f.write('\n'.join(ids))
        print(f'  {split_name}: {len(ids)} scenes -> {list_path}')

    # --- create info pkl files ---
    print()
    for split_name, ids in splits.items():
        create_info_pkl(args.output_dir, split_name, ids)

    print('\nDone!  Next steps:')
    print('  1. Check the output directory:')
    print(f'       {args.output_dir}/')
    print('  2. Start training:')
    print('       python tools/train.py configs/train_custom_forest.py')


if __name__ == '__main__':
    main()
