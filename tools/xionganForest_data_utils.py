# Copyright (c) OpenMMLab. All rights reserved.
import os
from concurrent import futures as futures
from os import path as osp

import mmengine
import numpy as np


class XionganForestData(object):
    """XionganForest data.
    Generate xionganForest infos for converter_xionganForest.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
        save_path (str, optional): Output directory.
    """

    def __init__(self, root_path, split='train', save_path=None):
        self.root_dir = root_path
        self.save_path = root_path if save_path is None else save_path
        self.split = split
        self.split_dir = osp.join(root_path)
        
        self.classes = [
                'tree'
            ]
        self.cat_ids = np.array([
                1
            ])

        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids2class = {
            treeid: i
            for i, treeid in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data',
                              f'{split}_list.txt')
        mmengine.check_file_exist(split_file)
        self.sample_id_list = mmengine.list_from_file(split_file)
        self.test_mode = (split == 'test')

    def __len__(self):
        return len(self.sample_id_list)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """Get data infos.

        This method gets information from the raw data.

        Args:
            num_workers (int, optional): Number of threads to be used.
                Default: 4.
            has_label (bool, optional): Whether the data has label.
                Default: True.
            sample_id_list (list[int], optional): Index list of the sample.
                Default: None.

        Returns:
            infos (list[dict]): Information of the raw data.
        """

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 3, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            pts_path = osp.join(self.root_dir, 'points',
                                f'{sample_idx}.bin')
            pts_instance_mask_path = osp.join(
                self.root_dir, 'instance_mask',
                f'{sample_idx}.bin')
            pts_semantic_mask_path = osp.join(
                self.root_dir, 'semantic_mask',
                f'{sample_idx}.bin')

            for fpath in [pts_path, pts_instance_mask_path,
                          pts_semantic_mask_path]:
                if not osp.isfile(fpath):
                    print(f'  [SKIP] {sample_idx}: {fpath} not found')
                    return None

            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')
            info['pts_instance_mask_path'] = osp.join(
                'instance_mask', f'{sample_idx}.bin')
            info['pts_semantic_mask_path'] = osp.join(
                'semantic_mask', f'{sample_idx}.bin')

            if has_label:
                annotations = {}
                axis_align_matrix = np.eye(4)

                pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 3)
                sem = np.fromfile(pts_semantic_mask_path, dtype=np.int64)
                ins = np.fromfile(pts_instance_mask_path, dtype=np.int64)

                bg_sem = np.array([0])
                valid_mask = ~np.isin(sem, bg_sem)
                valid_pts = pts[valid_mask]
                valid_ins = ins[valid_mask]

                unique_instance_ids = np.unique(valid_ins)
                unique_instance_ids = unique_instance_ids[unique_instance_ids != 0]
                num_instances = len(unique_instance_ids)

                instance_bboxes = np.zeros((num_instances, 7))
                for i, inst_id in enumerate(unique_instance_ids):
                    mask = valid_ins == inst_id
                    inst_pts = valid_pts[mask, :3]
                    if inst_pts.shape[0] == 0:
                        continue
                    min_pts = inst_pts.min(axis=0)
                    max_pts = inst_pts.max(axis=0)
                    instance_bboxes[i, :3] = (min_pts + max_pts) / 2
                    instance_bboxes[i, 3:6] = max_pts - min_pts
                    instance_bboxes[i, 6] = 1

                annotations['gt_num'] = num_instances
                if num_instances != 0:
                    aligned_box = instance_bboxes[:, :-1]
                    classes = instance_bboxes[:, -1]
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(num_instances)
                    ])
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = aligned_box[:, :3]
                    annotations['unaligned_dimensions'] = aligned_box[:, 3:6]
                    annotations[
                        'unaligned_gt_boxes_upright_depth'] = aligned_box
                    annotations['index'] = np.arange(
                        num_instances, dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(num_instances)
                    ])
                annotations['axis_align_matrix'] = axis_align_matrix
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None \
            else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return [i for i in infos if i is not None]
