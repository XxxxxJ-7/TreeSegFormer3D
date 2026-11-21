"""
Count treeIDs across all LAS/LAZ files in a folder.

Each LAS file has its own independent treeID numbering (starting from 0),
so counts are summed across files rather than deduplicated.

Usage:
    python tools/count_treeIDs.py /path/to/las_folder
    python tools/count_treeIDs.py /path/to/las_folder --recursive
    python tools/count_treeIDs.py /path/to/las_folder --verbose
    python tools/count_treeIDs.py data/xionganForest/test_data
    python tools/count_treeIDs.py data/xionganForest/test_data -v
"""

import argparse
import sys
from pathlib import Path

import laspy
import numpy as np


def find_trees_in_las(las_path: Path) -> np.ndarray:
    """Read a LAS/LAZ file and return the treeID array."""
    las = laspy.read(str(las_path))

    field_candidates = [
        'treeID', 'TreeID', 'tree_id', 'treeId', 'TreeId', 'treeid',
        'TREEID', 'TREE_ID',
    ]
    tree_id = None
    for name in field_candidates:
        try:
            tree_id = np.asarray(getattr(las, name), dtype=np.int64)
            break
        except AttributeError:
            continue

    if tree_id is None:
        dims = list(las.point_format.dimension_names)
        raise ValueError(
            f'Cannot find tree ID field in {las_path}. '
            f'Tried: {field_candidates}. '
            f'Available dimensions: {dims}'
        )

    return tree_id


def main():
    parser = argparse.ArgumentParser(
        description='Count treeIDs across all LAS/LAZ files in a folder (summed, not deduplicated).'
    )
    parser.add_argument(
        'folder', type=str,
        help='Path to the folder containing LAS/LAZ files.'
    )
    parser.add_argument(
        '--recursive', '-r', action='store_true',
        help='Recursively search subdirectories for LAS/LAZ files.'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print per-file statistics.'
    )
    parser.add_argument(
        '--show-ids', '-s', action='store_true',
        help='Print the full sorted list of unique tree IDs.'
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f'Error: "{folder}" is not a valid directory.')
        sys.exit(1)

    # Find all LAS/LAZ files
    patterns = ['*.las', '*.LAS', '*.laz', '*.LAZ']
    if args.recursive:
        las_files = []
        for pattern in patterns:
            las_files.extend(folder.rglob(pattern))
    else:
        las_files = []
        for pattern in patterns:
            las_files.extend(folder.glob(pattern))

    las_files = sorted(set(las_files))

    if not las_files:
        print(f'No LAS/LAZ files found in "{folder}".')
        sys.exit(0)

    print(f'Found {len(las_files)} LAS/LAZ file(s) in "{folder}".\n')

    total_points = 0
    total_trees = 0       # sum of tree counts across all files (not deduplicated)
    file_stats = []

    for las_path in las_files:
        try:
            tree_id = find_trees_in_las(las_path)
        except Exception as e:
            print(f'Skipping {las_path.name}: {e}')
            continue

        unique_in_file = set(np.unique(tree_id))
        trees_in_file = len(unique_in_file - {0})   # unique treeIDs excluding 0 (ground)

        file_stats.append((
            las_path.name,
            len(tree_id),
            trees_in_file,
        ))

        total_points += len(tree_id)
        total_trees += trees_in_file

    if args.verbose:
        header = f'{"File":<40} {"Total Points":>14}     {"Trees":>8}'
        print(header)
        print('-' * len(header))
        for name, points, trees in file_stats:
            print(f'{name:<40} {points:>14,}     {trees:>8,}')
        print()

    # Summary
    print('=' * 50)
    print('  SUMMARY')
    print('=' * 50)
    print(f'  Total LAS/LAZ files processed : {len(file_stats)}')
    print(f'  Total points across all files : {total_points:,}')
    print(f'  Total trees (summed, excl. 0): {total_trees}')

    if args.show_ids:
        print()
        for name, _, _ in file_stats:
            las_path = folder / name
            if las_path.exists():
                tree_id = find_trees_in_las(las_path)
                ids = sorted(set(np.unique(tree_id)) - {0})
                print(f'  {name}:')
                col_width = max(len(str(x)) for x in ids) + 2 if ids else 4
                per_row = max(1, 80 // col_width)
                for i in range(0, len(ids), per_row):
                    chunk = ids[i:i + per_row]
                    print('    ' + ''.join(f'{x:<{col_width}}' for x in chunk))
                print()


if __name__ == '__main__':
    main()
