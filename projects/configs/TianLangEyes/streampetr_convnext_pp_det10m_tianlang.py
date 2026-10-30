# 自有数据推理配置。结构和 det10m 训练配置一致，输入改成 512×640。
# 四路环视带畸变进模型，IPM 用内外参把特征投到 BEV。
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
# DINOv3 ConvNeXt-Tiny, backbone weights frozen.
# task: 'det' | 'occ' | 'both'
task = 'both'
# 图像交叉注意力。False 用原来的全连接注意力，True 用可变形注意力。
# det_from_view 为 True 时检测不走这条，改看下面 PlaneViewTransformer 的 BEV。
use_img_deform_attn = True
# 检测不从原图特征取 token，改为接 img_view_transformer 输出的相机 BEV。
det_from_view = True
_view_channels = 64
# 单视觉：use_camera=True, use_lidar=False。单雷达：反过来。
# 两路都开时，检测看雷达 BEV，占用把雷达 BEV 加到相机 BEV 上。
use_camera = True
use_lidar = False
# 图像骨干和雷达骨干分开冻。True 时该骨干权重不更新，并保持 eval。
freeze_img_backbone = True
freeze_lidar_backbone = False
# 检测跨帧 memory。False 时 query 只看当前帧，不传播历史框，也不写 memory。
# True 时第一层只用当前帧，refine 后用历史框换掉低分 query，之后每层把预测中心写回采样点。
use_det_temporal = False
data_root = './data/nuscenes/'
occ_root = data_root + 'gts/'
demo_data_root = '/mnt/data/nas/datasets/test_my_data'
demo_calib = '/mnt/data/nas/datasets/test_my_data/calibration.yaml'
# 由光轴定的顺序：前、左、右、后。目录里没有 left/right 两路鱼眼。
demo_cameras = ['cam3', 'cam2', 'cam0', 'cam1']
demo_lidars = ['front_up', 'back_down']

# 检测框前后左右各 9.6 m。0.1 m 柱子画布是 192×192。
# SECOND 第一层 stride 2，FPN 对齐到 96×96，输出每格 0.2 m。
# position_range 比检测框四周再扩 10 m，给图像位置编码留白。
# 占用单独用 0.1 m 格子：xy 同样 ±9.6 m（192×192），z 从 -2 m 到 1 m（30 层）。
point_cloud_range = [-9.6, -9.6, -3.5, 9.6, 9.6, 2.5] # 3d检测范围
position_range = [-19.6, -19.6, -5.5, 19.6, 19.6, 4.5] # 原图采特征时，用更大范围
lidar_pc_range = [-9.6, -9.6, -3.0, 9.6, 9.6, 3.0] # lidar backbone范围
lidar_voxel_size = [0.1, 0.1, 6.0] # lidar pillar size
_pillar_nx = int(round((lidar_pc_range[3] - lidar_pc_range[0]) / lidar_voxel_size[0]))
_pillar_ny = int(round((lidar_pc_range[4] - lidar_pc_range[1]) / lidar_voxel_size[1]))
occ_range = [-9.6, -9.6, -2.5, 9.6, 9.6, 0.5]
occ_voxel = [0.1, 0.1, 0.1]
_occ_dz = int(round((occ_range[5] - occ_range[2]) / occ_voxel[2]))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# 4 类：小汽车单独留下。卡车、工程车、公交、挂车合成其他车。
# 摩托车和自行车合成两轮车。路障和交通锥不在映射里，训练时丢掉。
class_names = ['car', 'other_vehicle', 'cycle', 'pedestrian']
class_merge = dict(
    truck='other_vehicle',
    construction_vehicle='other_vehicle',
    bus='other_vehicle',
    trailer='other_vehicle',
    motorcycle='cycle',
    bicycle='cycle',
)
num_gpus = 1
batch_size = 32
num_iters_per_epoch = 60000 // (num_gpus * batch_size)
num_epochs = 24
queue_length = 1
num_frame_losses = 1
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
    'img_timestamp', 'ego_pose', 'ego_pose_inv'
]
input_modality = dict(
    use_lidar=use_lidar, use_camera=use_camera,
    use_radar=False, use_map=False, use_external=True)

model = dict(
    type='StreamPETRFusion',
    tasks=task,
    use_lidar=use_lidar,
    use_camera=use_camera,
    freeze_img_backbone=freeze_img_backbone,
    freeze_lidar_backbone=freeze_lidar_backbone,
    det_from_view=det_from_view,
    num_frame_head_grads=1,
    num_frame_backbone_grads=1,
    num_frame_losses=1,
    use_grid_mask=False,
    stride=16,
    position_level=0,
    lidar_pc_range=lidar_pc_range,
    lidar_voxel_size=lidar_voxel_size,
    max_voxels=_pillar_nx * _pillar_ny,
    max_points=32,
    img_backbone=dict(
        type='ConvNeXt',
        arch='tiny',
        out_indices=(2, 3),
        frozen=freeze_img_backbone,
        pretrained='ckpts/dinov3_convnext_tiny_lvd1689m.safetensors'),
    img_neck=dict(type='CPFPN', in_channels=[384, 768], out_channels=256, num_outs=2),
    pts_backbone=dict(type='PointPillars', ny=_pillar_ny, nx=_pillar_nx),
    img_view_transformer=dict(
        type='PlaneViewTransformer',
        grid_config=dict(
            x=[occ_range[0], occ_range[3], occ_voxel[0]],
            y=[occ_range[1], occ_range[4], occ_voxel[1]]),
        input_size=(512, 640),
        in_channels=256,
        out_channels=_view_channels,
        downsample=16,
        bev_planes=(-1.5, -0.5, 0.5)),
        # bev_planes=(-1.0, 0, 1.0)),
    bev_backbone=dict(
        type='UNet',
        in_channels=_view_channels,
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
        Dz=_occ_dz,
        use_mask=False,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        perception_range=occ_range,
        occ_label_range=occ_range,
        xy_voxel_size=occ_voxel[0],
        loss_occ=dict(
            type='CustomFocalLoss', use_sigmoid=True, loss_weight=1.0)),
    pts_bbox_head=dict(
        type='StreamPETRLidarHead',
        lidar_in_channels=384,
        lidar_num_points=4,
        use_img_bev=det_from_view,
        img_bev_channels=_view_channels,
        num_classes=len(class_names),
        in_channels=256,
        num_query=384,
        memory_len=1024,
        topk_proposals=128,
        num_propagated=128,
        use_temporal=use_det_temporal,
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
            num_classes=len(class_names)),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[_pillar_nx, _pillar_ny, 1],
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'CustomNuScenesDataset'
# 高×宽。1920×1536 正好缩到 1/3，不裁切。IPM 的 input_size 必须和这张图一致。
ida_aug_conf = dict(
    resize_lim=(0.38, 0.55), final_dim=(512, 640), bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0), H=900, W=1600, rand_flip=True)
det_keys = ['points'] if use_lidar else []
occ_keys = ['voxel_semantics', 'mask_camera'] if task in ('occ', 'both') else []
train_pipeline = [
    dict(type='LoadMultiViewImageAndPoints', to_float32=False,
         load_points=use_lidar,
         coord_type='LIDAR', load_dim=5, use_dim=4)]
if task in ('occ', 'both'):
    train_pipeline.append(dict(
        type='LoadOcc3DFromFile', occ_root=occ_root,
        target_range=occ_range, target_voxel=occ_voxel))
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
    dict(type='LoadMultiViewImageAndPoints', to_float32=False,
         load_points=use_lidar,
         coord_type='LIDAR', load_dim=5, use_dim=4),
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
        class_merge=class_merge,
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
             classes=class_names, class_merge=class_merge, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline,
              collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length,
              ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
              classes=class_names, class_merge=class_merge, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing', warmup='linear', warmup_iters=500,
    warmup_ratio=1.0 / 3, min_lr_ratio=2e-4)
evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)
find_unused_parameters = True
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = 'work_dirs/streampetr_convnext_pp_det10m_nus/latest.pth'
resume_from = None
file_client_args = dict(backend='disk')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
