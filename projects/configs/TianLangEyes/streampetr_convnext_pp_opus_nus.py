# 检测、数据、优化器和图像骨干与 streampetr_convnext_pp_lss_nus.py 相同。
# 占用换成 OPUS-T：稀疏 query 预测占据点和类别，不再走平面投影、UNet 和 BEVOCCHead2D。
# 雷达按 OPUS fusion 接入：PointPillars BEV 投到 256 维后，和图像采样特征拼接再混合。
# 官方 OPUS-T 是 8 帧、4 级 FPN。这里保持现有单帧和 CPFPN 的 2 级输出。
# pts_range 是 PointPillars 画布，和基类 lidar_pc_range 一致。
_base_ = ['./streampetr_convnext_pp_lss_nus.py']

opus_pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
opus_voxel_size = [0.4, 0.4, 0.4]
opus_num_layers = 6
opus_num_levels = 2
opus_num_query = 600
opus_num_points = 4
opus_num_refines = [1, 4, 16, 32, 64, 128]
opus_pts_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

model = dict(
    img_view_transformer=None,
    bev_backbone=None,
    occ_head=dict(
        _delete_=True,
        type='OPUSV1Head',
        num_classes=17,
        in_channels=256,
        num_query=opus_num_query,
        pc_range=opus_pc_range,
        voxel_size=opus_voxel_size,
        transformer=dict(
            type='OPUSV1Transformer',
            embed_dims=256,
            num_frames=1,
            num_points=opus_num_points,
            num_layers=opus_num_layers,
            num_levels=opus_num_levels,
            num_classes=17,
            num_refines=opus_num_refines,
            scales=[0.5],
            pc_range=opus_pc_range,
            with_pts=True,
            pts_range=opus_pts_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_pts=dict(type='SmoothL1Loss', beta=0.2, loss_weight=0.5),
        test_cfg=dict(score_thr=0.5, padding=True)),
    train_cfg=dict(
        occ=dict(
            cls_weights=[
                10, 5, 10, 5, 5, 10, 10, 5, 10, 5, 5, 1, 5, 1, 1, 2, 1])))
