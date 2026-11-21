"""Convert the 19 original LAS/LAZ plots into MMDetection3D binary files."""
import argparse
import json
from pathlib import Path
import pickle

import laspy
import numpy as np

from .protocol import SPLITS, TEST, map_labels, plot_id


def discover(raw_root):
    found = {}
    for path in sorted(Path(raw_root).rglob('*')):
        if path.suffix.lower() not in ('.las', '.laz'):
            continue
        ident = plot_id(path)
        if ident in found:
            raise ValueError(f'Duplicate plot {ident}: {found[ident]} and {path}')
        found[ident] = path
    expected = {i for ids in SPLITS.values() for i in ids}
    if set(found) != expected:
        raise ValueError(f'Incomplete/unexpected dataset: missing={sorted(expected-set(found))}, '
                         f'extra={sorted(set(found)-expected)}')
    return found


def convert_plot(path, out_root, expected_tree_ids=None):
    path, out_root = Path(path), Path(out_root)
    ident = plot_id(path)
    cloud = laspy.read(path)
    if 'tree_index' not in set(cloud.point_format.dimension_names):
        raise ValueError(f'{path}: missing tree_index')
    keep, semantic, instance = map_labels(cloud.classification, cloud['tree_index'])
    xyz = np.asarray(cloud.xyz[keep], dtype=np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError(f'{path}: empty/nonfinite point cloud')
    gt_ids = set(np.unique(instance)) - {0}
    if expected_tree_ids is not None and gt_ids != set(expected_tree_ids):
        raise ValueError(f'{path}: tree IDs disagree with official plot_data.yaml')
    # Shift in float64 BEFORE converting to float32 (avoid geospatial precision loss).
    offset = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].min()])
    xyz = (xyz - offset).astype('<f4')
    instances = []
    for tree_id in sorted(gt_ids):
        tree_xyz = xyz[instance == tree_id]
        low, high = tree_xyz.min(axis=0), tree_xyz.max(axis=0)
        instances.append(dict(
            bbox_3d=np.r_[((low + high) / 2), (high - low)].tolist(),
            bbox_label_3d=1))
    filename = f'plot_{ident}.bin'
    for subdir in ('points', 'semantic_mask', 'instance_mask', 'source_indices'):
        (out_root / subdir).mkdir(parents=True, exist_ok=True)
    xyz.tofile(out_root / 'points' / filename)
    semantic.astype('<i8').tofile(out_root / 'semantic_mask' / filename)
    instance.astype('<i8').tofile(out_root / 'instance_mask' / filename)
    np.flatnonzero(keep).astype('<i8').tofile(out_root / 'source_indices' / filename)
    info = dict(sample_idx=ident, plot_id=ident,
                lidar_points=dict(num_pts_feats=3, lidar_path=filename),
                pts_semantic_mask_path=filename, pts_instance_mask_path=filename,
                axis_align_matrix=np.eye(4).tolist(), instances=instances,
                xyz_offset=offset.tolist())
    audit = dict(plot_id=ident, source=str(path.resolve()), source_points=len(cloud.points),
                 retained_points=len(xyz), excluded_class5=int((~keep).sum()),
                 trees=len(gt_ids), xyz_offset=offset.tolist())
    return info, audit


def prepare(raw_root, out_root):
    import yaml

    raw_root, out_root = Path(raw_root), Path(out_root)
    found = discover(raw_root)  # Check all 19 before writing any split index.
    with (raw_root / 'plot_data.yaml').open(encoding='utf-8') as stream:
        metadata = yaml.safe_load(stream)
    infos, audits = {}, []
    for ident, path in found.items():
        meta = metadata[ident]
        if bool(meta['test']) != (ident in TEST):
            raise ValueError(f'Official split metadata conflict for plot {ident}')
        print(f'Preparing plot {ident} ...', flush=True)
        info, audit = convert_plot(path, out_root, meta['trees'].keys())
        if audit['trees'] != meta['n_trees']['all']:
            raise ValueError(f'Tree count mismatch: {ident}')
        infos[ident] = info
        audits.append(audit)
    for split, ids in SPLITS.items():
        result = dict(metainfo=dict(categories={'other': 0, 'tree': 1},
                                    dataset='FGI-EMIT', info_version='1.1'),
                      data_list=[infos[i] for i in ids])
        with (out_root / f'fgi_emit_infos_{split}.pkl').open('wb') as stream:
            pickle.dump(result, stream)
    manifest = dict(record='https://zenodo.org/records/19351234',
                    splits=SPLITS, features='XYZ only', voxelized_before_evaluation=False,
                    labels={'other': 0, 'tree': 1, 'excluded_source_classes': [5]}, plots=audits)
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared 10 training / 3 validation / 6 test plots at {out_root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('data/FGI_EMIT/raw'))
    parser.add_argument('--out-root', type=Path, default=Path('data/FGI_EMIT/processed'))
    args = parser.parse_args()
    prepare(args.raw_root, args.out_root)


if __name__ == '__main__':
    main()
