import numpy as np

from mmdet3d.datasets.scannet_dataset import ScanNetDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class CustomForestDataset(ScanNetDataset):
    """Forest point cloud dataset with 2 classes: ground and tree.

    Semantic classes:
        0 - ground (stuff)
        1 - tree   (thing, each tree is one instance)
    """

    METAINFO = {
        'classes': ('ground', 'tree'),
        'palette': [[34, 139, 34], [139, 69, 19]],
        'seg_valid_class_ids': (0, 1),
        'seg_all_class_ids': (0, 1),
    }

    def get_scene_idxs(self, *args, **kwargs):
        return np.arange(len(self)).astype(np.int32)

    def parse_data_info(self, info: dict) -> dict:
        info = super().parse_data_info(info)
        return info
