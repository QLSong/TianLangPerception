from typing import Literal


_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# DINOv3 ConvNeXt-Tiny, backbone weights frozen.
# task: 'det' | 'occ' | 'both'
task = 'both'
# 图像交叉注意力。False 用原来的全连接注意力，True 用可变形注意力。
use_img_deform_attn = True
# 关掉后检测和占用都不再读点云。打开时两个任务共用 PointPillars BEV。
use_lidar = True
data_root = './data/nuscenes/'
occ_root = data_root + 'gts/'

# 检测框、PointPillars、LSS 的 xy 都用 point_cloud_range。
# pillar 0.2 m，画布 512×512。SECOND+FPN 输出格子是 pillar 的 2 倍（0.4 m，256×256）。
# LSS 用同一 xy 和 0.4 m。Occ3D 标签仍是 ±40 m 的 200×200×16，损失只监督这块。
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
position_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
lidar_pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
lidar_voxel_size = [0.2, 0.2, 8.0]
_pillar_nx = int(round((lidar_pc_range[3] - lidar_pc_range[0]) / lidar_voxel_size[0]))
_pillar_ny = int(round((lidar_pc_range[4] - lidar_pc_range[1]) / lidar_voxel_size[1]))
_bev_step = lidar_voxel_size[0] * 2
occ_label_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_gpus = 1
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
queue_length = 1
num_frame_losses = 1
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
    'img_timestamp', 'ego_pose', 'ego_pose_inv'
]
input_modality = dict(
    use_lidar=use_lidar, use_camera=True,
    use_radar=False, use_map=False, use_external=True)

model = dict(
    type='StreamPETRFusion',
    tasks=task,
    use_lidar=use_lidar,
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    use_grid_mask=False,
    stride=16,
    position_level=0,
    lidar_pc_range=lidar_pc_range,
    lidar_voxel_size=lidar_voxel_size,
    max_voxels=12000,
    max_points=32,
    img_backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(2, 3),
        frozen=True,
        pretrained='ckpts/dinov3_convnext_tiny_lvd1689m.safetensors'),
    img_neck=dict(type='CPFPN', in_channels=[384, 768], out_channels=256, num_outs=2),
    pts_backbone=dict(type='PointPillars', ny=_pillar_ny, nx=_pillar_nx),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=dict(
            x=[lidar_pc_range[0], lidar_pc_range[3], _bev_step],
            y=[lidar_pc_range[1], lidar_pc_range[4], _bev_step],
            z=[-1.0, 5.4, 6.4],
            depth=[1.0, 45.0, 0.5]),
        input_size=(256, 704),
        in_channels=256,
        out_channels=64,
        sid=False,
        collapse_z=True,
        downsample=16),
    bev_backbone=dict(
        type='UNet',
        in_channels=64,
        base_channels=64,
        num_stages=4,
        strides=(1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2),
        dec_num_convs=(2, 2, 2),
        downsamples=(True, True, True),
        enc_dilations=(1, 1, 1, 1),
        dec_dilations=(1, 1, 1),
        upsample_cfg=dict(type='InterpConv')),
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=256,
        out_dim=256,
        Dz=16,
        use_mask=False,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        perception_range=lidar_pc_range,
        occ_label_range=occ_label_range,
        xy_voxel_size=_bev_step,
        loss_occ=dict(
            type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0)),
    pts_bbox_head=dict(
        type='StreamPETRLidarHead',
        lidar_in_channels=384,
        lidar_num_points=4,
        num_classes=10,
        in_channels=256,
        num_query=384,
        memory_len=1024,
        topk_proposals=128,
        num_propagated=128,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        LID=True,
        with_position=True,
        position_range=position_range,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=3,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                        dict(
                            type='ImageDeformCrossAttention', embed_dims=256, num_heads=8,
                            num_points=4, num_cams=6, dropout=0.1,
                            pc_range=point_cloud_range)
                        if use_img_deform_attn else
                        dict(type='PETRMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=False,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=position_range,
            pc_range=point_cloud_range,
            max_num=128,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55), final_dim=(256, 704), bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0), H=900, W=1600, rand_flip=True)
det_keys = ['points'] if use_lidar else []
occ_keys = ['voxel_semantics', 'mask_camera'] if task in ('occ', 'both') else []
train_pipeline = [
    dict(type='LoadMultiViewImageAndPoints', to_float32=True,
         load_points=use_lidar,
         coord_type='LIDAR', load_dim=5, use_dim=4)]
if task in ('occ', 'both'):
    train_pipeline.append(dict(type='LoadOcc3DFromFile', occ_root=occ_root))
train_pipeline += [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=1),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
               'centers2d', 'depths', 'prev_exists'] + det_keys + occ_keys + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token',
                    'gt_bboxes_3d', 'gt_labels_3d')),
]
test_pipeline = [
    dict(type='LoadMultiViewImageAndPoints', to_float32=True,
         load_points=use_lidar,
         coord_type='LIDAR', load_dim=5, use_dim=4)]
test_pipeline += [
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=1),
    dict(type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PETRFormatBundle3D', collect_keys=collect_keys,
                  class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['img'] + det_keys + collect_keys,
                  meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                             'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                             'img_norm_cfg', 'scene_token')),
         ]),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    prefetch_factor=2,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_temporal_infos_train.pkl',
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline,
             collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length,
             ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
             classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline,
              collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length,
              ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing', warmup='linear', warmup_iters=500,
    warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
_eval_interval = num_iters_per_epoch * num_epochs if task != 'occ' else num_iters_per_epoch * num_epochs + 1
evaluation = dict(interval=_eval_interval, pipeline=test_pipeline)
find_unused_parameters = True
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None
