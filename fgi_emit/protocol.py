"""Dataset protocol and CPU utilities, independent of the training stack."""
from pathlib import Path
import re

import numpy as np

RECORD = '19351234'
OFFICIAL_TRAIN = (1001, 1003, 1005, 1009, 1010, 1013, 1019, 1020, 1022,
                  1023, 1024, 1027, 1031)
VAL = (1019, 1022, 1031)
TEST = (1002, 1004, 1008, 1012, 1018, 1028)
TRAIN = tuple(i for i in OFFICIAL_TRAIN if i not in VAL)
SPLITS = dict(train=TRAIN, val=VAL, test=TEST)
FILES = {
    'training.zip': 'b1c5ee7e9c166dd7ff377539944e5002',
    'test.zip': 'c67b80c1e7f8a2ea28f54ed364aceea2',
    'accuracy.zip': '368c1b7228b1ca63ae16eae3dddc8388',
    'metadata.yaml': 'a42b36c6f11e7c571b29d9006129c458',
    'plot_data.yaml': '0e5f69c28be31374bb3cf8faa03bded0',
}


def plot_id(path):
    match = re.fullmatch(r'plot_(\d+)(?:_.*)?', Path(path).stem)
    if not match:
        raise ValueError(f'Expected plot_<ID>[_suffix].las/laz: {path}')
    return int(match.group(1))


def map_labels(classification, tree_index):
    """Exclude class 5; keep 2/3/4 as background. Never infer GT from geometry."""
    cls = np.asarray(classification, dtype=np.int64)
    inst = np.asarray(tree_index, dtype=np.int64)
    if cls.shape != inst.shape or cls.ndim != 1:
        raise ValueError('Classification and instance arrays must be aligned vectors.')
    if not np.isin(cls, np.arange(6)).all() or (inst < 0).any():
        raise ValueError('Unexpected FGI-EMIT class or negative instance ID.')
    keep = cls != 5
    if not np.array_equal(cls[keep] == 1, inst[keep] > 0):
        raise ValueError('Tree semantics and tree_index disagree; inspect source labels.')
    return keep, (cls[keep] == 1).astype(np.int64), inst[keep]


def voxel_representatives(xyz, size):
    """One deterministic representative per voxel, plus full-resolution mapping."""
    xyz = np.asarray(xyz)
    if size <= 0 or xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Expected nonempty XYZ and positive voxel size.')
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite coordinates.')
    cells = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    _, representative, inverse = np.unique(
        cells, axis=0, return_index=True, return_inverse=True)
    return representative, inverse


def tile_centers(xyz, radius, stride):
    if radius <= 0 or not 0 < stride <= radius:
        raise ValueError('Require 0 < stride <= radius for full XY coverage.')
    low, high = np.min(xyz[:, :2], axis=0), np.max(xyz[:, :2], axis=0)
    for x in np.arange(low[0], high[0] + stride, stride):
        for y in np.arange(low[1], high[1] + stride, stride):
            yield np.array([x, y])


class InstanceMerger:
    """Streaming tile merging with local-overlap IoU and score-based ownership.

    IDs start at 1; 0 is background. Compare each candidate to existing labels
    restricted to the SAME tile so a clipped crown can match its larger union.
    This shared postprocessor is independent of GT and of the model family.
    """
    def __init__(self, n_points, iou_threshold=0.5):
        if not 0 < iou_threshold <= 1:
            raise ValueError('Invalid merge IoU.')
        self.labels = np.zeros(n_points, dtype=np.int64)
        self.point_scores = np.full(n_points, -np.inf, dtype=np.float32)
        self.scores = [0.0]
        self.iou_threshold = iou_threshold

    def add(self, tile_indices, mask, score):
        tile_indices = np.asarray(tile_indices)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != tile_indices.shape or not np.isfinite(score):
            raise ValueError('Invalid tile mask or score.')
        candidate = tile_indices[mask]
        if not len(candidate):
            return
        present = self.labels[tile_indices]
        ids, intersect = np.unique(present[mask], return_counts=True)
        best_id, best_iou = 0, 0.0
        for label, count in zip(ids, intersect):
            if label == 0:
                continue
            union = len(candidate) + np.count_nonzero(present == label) - count
            overlap = count / union
            if overlap > best_iou:
                best_id, best_iou = int(label), overlap
        if best_iou < self.iou_threshold:
            best_id = len(self.scores)
            self.scores.append(float(score))
        else:
            self.scores[best_id] = max(self.scores[best_id], float(score))
        update = candidate[score > self.point_scores[candidate]]
        self.labels[update] = best_id
        self.point_scores[update] = score

    def finish(self, semantic):
        labels = self.labels.copy()
        labels[np.asarray(semantic) == 0] = 0
        # A single confidence per final instance is required by official AP.
        scores = np.asarray(self.scores, dtype=np.float32)[labels]
        return labels, scores
