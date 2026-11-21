_base_ = ['mmdet3d::_base_/default_runtime.py']

custom_imports = dict(imports=['fgi_emit.integration'])
data_root = 'data/FGI_EMIT/processed/'
raw_root = 'data/FGI_EMIT/raw/'
# FGI-EMIT paper, Appendix C.4: ForestFormer3D used 12 m cylinders.
# The same crop is shared by all three models for a controlled comparison.
radius = 12.0
voxel_size = 0.2
max_points = 640000

data_prefix = dict(pts='points', pts_instance_mask='instance_mask',
                   pts_semantic_mask='semantic_mask')
load_points = dict(type='LoadPointsFromFile', coord_type='DEPTH',
                   shift_height=False, use_color=False, load_dim=3,
                   use_dim=[0, 1, 2])
load_labels = dict(type='LoadAnnotations3D', with_bbox_3d=False,
                   with_label_3d=False, with_mask_3d=True, with_seg_3d=True)
meta_keys = ('lidar_path', 'sample_idx', 'plot_id', 'xyz_offset')

train_pipeline = [
    load_points,
    load_labels,
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5,
         flip_ratio_bev_vertical=0.0),
    dict(type='GlobalRotScaleTrans', rot_range=[-3.1415926, 3.1415926],
         scale_ratio_range=[0.8, 1.2], translation_std=[0.1, 0.1, 0.1],
         shift_height=False),
    dict(type='FGIEMITTrainSample', radius=radius, voxel_size=voxel_size,
         max_points=max_points),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
               'pts_instance_mask', 'ratio_inspoint', 'vote_label',
               'instance_mask'], meta_keys=meta_keys),
]
test_pipeline = [
    load_points,
    load_labels,
    dict(type='FGIEMITMakeWritable'),
    dict(type='Pack3DDetInputs_',
         keys=['points', 'gt_labels_3d', 'pts_semantic_mask',
               'pts_instance_mask', 'instance_mask'], meta_keys=meta_keys),
]

train_dataloader = dict(
    batch_size=2, num_workers=4, pin_memory=True, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='FGIEMITDataset', data_root=data_root,
                 ann_file='fgi_emit_infos_train.pkl', data_prefix=data_prefix,
                 pipeline=train_pipeline, filter_empty_gt=True,
                 box_type_3d='Depth', backend_args=None))
val_dataloader = dict(
    # Full plots contain millions of points. Loading them in the main process
    # avoids Docker /dev/shm exhaustion when a worker transfers the batch.
    batch_size=1, num_workers=0, pin_memory=True, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='FGIEMITDataset', data_root=data_root,
                 ann_file='fgi_emit_infos_val.pkl', data_prefix=data_prefix,
                 pipeline=test_pipeline, test_mode=True, box_type_3d='Depth',
                 backend_args=None))
test_dataloader = dict(
    batch_size=1, num_workers=0, pin_memory=True, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='FGIEMITDataset', data_root=data_root,
                 ann_file='fgi_emit_infos_test.pkl', data_prefix=data_prefix,
                 pipeline=test_pipeline, test_mode=True, box_type_3d='Depth',
                 backend_args=None))

val_evaluator = dict(type='FGIEMITOfficialMetric', split='val',
                     pred_root='work_dirs/fgi_emit/predictions',
                     data_root=data_root, raw_root=raw_root)
test_evaluator = dict(type='FGIEMITOfficialMetric', split='test',
                      pred_root='work_dirs/fgi_emit/predictions',
                      data_root=data_root, raw_root=raw_root)

optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
                     clip_grad=dict(max_norm=10, norm_type=2))
# Paper setting: 10 training plots / batch size 2 x 6,500 epochs
# = 32,500 optimizer steps.
param_scheduler = dict(type='PolyLR', begin=0, end=32500, power=0.9,
                       eta_min=1e-6, by_epoch=False)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6500, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
custom_hooks = [dict(type='SetTrainEpochHook'),
                dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=100, max_keep_ckpts=3,
                    save_optimizer=True, save_best='fgi_emit/f1', rule='greater'),
    logger=dict(type='LoggerHook', interval=10),
    visualization=dict(type='Det3DVisualizationHook', draw=False))
vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends,
                  name='visualizer')
randomness = dict(seed=2026, deterministic=False)
find_unused_parameters = True
