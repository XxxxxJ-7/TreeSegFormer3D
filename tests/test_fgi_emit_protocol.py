import unittest
from pathlib import Path
import tempfile

import laspy
import numpy as np

from fgi_emit.protocol import (InstanceMerger, OFFICIAL_TRAIN, SPLITS, TEST,
                               VAL, map_labels, tile_centers,
                               voxel_representatives)
from fgi_emit.prepare import convert_plot


class FGIEmitProtocolTest(unittest.TestCase):
    def test_split_is_complete_and_disjoint(self):
        self.assertEqual(len(SPLITS['train']), 10)
        self.assertEqual(len(VAL), 3)
        self.assertEqual(len(TEST), 6)
        self.assertEqual(set(SPLITS['train']) | set(VAL), set(OFFICIAL_TRAIN))
        self.assertFalse(set(SPLITS['train']) & set(VAL))
        self.assertFalse(set(OFFICIAL_TRAIN) & set(TEST))

    def test_label_policy(self):
        keep, semantic, instance = map_labels(
            np.array([0, 1, 2, 3, 4, 5]), np.array([0, 11, 0, 0, 0, 0]))
        np.testing.assert_array_equal(keep, [1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(semantic, [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(instance, [0, 11, 0, 0, 0])
        with self.assertRaises(ValueError):
            map_labels(np.array([1]), np.array([0]))

    def test_voxel_mapping_is_deterministic(self):
        xyz = np.array([[0., 0., 0.], [.1, .1, 0.], [.3, 0., 0.]])
        representative, inverse = voxel_representatives(xyz, .2)
        np.testing.assert_array_equal(representative, [0, 2])
        np.testing.assert_array_equal(inverse, [0, 0, 1])

    def test_overlap_merging_and_confidence(self):
        merger = InstanceMerger(6, iou_threshold=.4)
        merger.add(np.array([0, 1, 2, 3]), np.array([0, 1, 1, 0]), .7)
        merger.add(np.array([1, 2, 3, 4]), np.array([1, 1, 0, 0]), .8)
        labels, scores = merger.finish(np.array([0, 1, 1, 0, 0, 0]))
        np.testing.assert_array_equal(labels, [0, 1, 1, 0, 0, 0])
        np.testing.assert_allclose(scores[1:3], .8)

    def test_tiling_covers_bounding_box(self):
        xyz = np.array([[0., 0., 0.], [10., 10., 1.]])
        centers = np.array(list(tile_centers(xyz, radius=6., stride=4.)))
        grid = np.array([[x, y] for x in np.linspace(0, 10, 21)
                        for y in np.linspace(0, 10, 21)])
        self.assertTrue((((grid[:, None] - centers) ** 2).sum(2) <= 36)
                        .any(1).all())

    def test_las_conversion_preserves_instances_and_precision(self):
        temp_root = Path('tmp')
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as folder:
            root = Path(folder)
            header = laspy.LasHeader(point_format=3, version='1.2')
            header.scales = np.array([.001, .001, .001])
            header.offsets = np.array([500000., 7000000., 0.])
            header.add_extra_dim(laspy.ExtraBytesParams(name='tree_index', type='u4'))
            cloud = laspy.LasData(header)
            cloud.x = [500000., 500000.1, 500001., 500002.]
            cloud.y = [7000000., 7000000.1, 7000001., 7000002.]
            cloud.z = [100., 101., 99., 102.]
            cloud.classification = [1, 1, 0, 5]
            cloud['tree_index'] = [7, 7, 0, 0]
            source = root / 'plot_1001.las'
            cloud.write(source)
            info, audit = convert_plot(source, root / 'processed', [7])
            points = np.fromfile(root / 'processed/points/plot_1001.bin',
                                 dtype='<f4').reshape(-1, 3)
            self.assertEqual(points.shape, (3, 3))
            self.assertEqual(audit['excluded_class5'], 1)
            self.assertEqual(len(info['instances']), 1)
            self.assertLess(np.abs(points[:, :2]).max(), 2.)


if __name__ == '__main__':
    unittest.main()
