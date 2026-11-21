_base_ = ['./_base_fgi_emit.py']

work_dir = 'work_dirs/fgi_emit/treesegformer3d'
val_evaluator = dict(pred_root='work_dirs/fgi_emit/treesegformer3d/predictions')
test_evaluator = dict(pred_root='work_dirs/fgi_emit/treesegformer3d/predictions')

model = dict(
    type='FGIEMITTreeSegFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=3, num_channels=32, voxel_size=0.2, num_classes=2,
    min_spatial_shape=128, stuff_classes=[0], thing_cls=[1], radius=12.0,
    prepare_epoch=50, query_point_num=300, score_th=0.4, chunk=20000,
    n_seeds=50, min_per_cluster=3, bcl_weight=0.5, bcl_margin=1.5,
    bcl_neighbor_radius=2.0, bcl_max_pairs=4096, bcl_max_neighbors=32,
    bcl_temperature=0.1,
    bcl_hard_neg_ratio=0.5,
    backbone=dict(type='SpConvUNet', num_planes=[32, 64, 96, 128, 160],
                  return_blocks=True),
    decoder=dict(type='ForAINetv2QueryDecoder_XAwarequery_V3', num_layers=6,
                 num_classes=1, num_instance_queries=0,
                 num_semantic_queries=2, num_instance_classes=2,
                 in_channels=32, d_model=256, num_heads=8, hidden_dim=1024,
                 dropout=0.0, activation_fn='gelu', iter_pred=True,
                 attn_mask=True, fix_attention=True, objectness_flag=True,
                 pos_enc_max_freq=10000.0, mlp_mask_hidden_dim=256,
                 checkpoint_mask_head=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion_XAwarequery', num_semantic_classes=2,
        sem_criterion=dict(type='S3DISSemanticCriterion', loss_weight=0.2),
        inst_criterion=dict(type='InstanceCriterionForAI_OneToManyMatch',
                            matcher=dict(type='One2ManyMatcher'),
                            loss_weight=[1.0, 1.0, 0.5],
                            fix_dice_loss_weight=True, iter_matcher=True,
                            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(topk_insts=300, inst_score_thr=0.4, pan_score_thr=0.0,
                  npoint_thr=10, obj_normalization=True,
                  obj_normalization_thr=0.01, sp_score_thr=0.15, nms=True,
                  matrix_nms_kernel='linear', num_sem_cls=2,
                  tile_stride=3.0, max_points=640000, merge_iou=0.3,
                  edge_margin=0.5,
                  inference_seed=2026))
