# 检测、数据、优化器和图像骨干与 streampetr_convnext_pp_lss_nus.py 相同。
# 占用换成 GaussianOcc。监督对齐 configs/nusc-sem-gs.txt 的 self：
# 前后帧光度重投影、视差平滑 0.001、二维语义交叉熵 0.02、不透明度熵 0.1、体素 TV 0.1。
# 不拿 Occ3D 体素标签做训练损失。Occ3D 只留给评测。
# 3D Gaussian 光栅化扩展没编译，深度和语义图用官方 prob 射线积分。
# 位姿用传感器外参的 cam_T_cam。官方 gt_pose=yes 用的 stage1 位姿网络这里没有。
# 图像仍是现有的 256×704，渲染分辨率取一半 128×352。
_base_ = ['./streampetr_convnext_pp_lss_nus.py']

model = dict(
    img_view_transformer=None,
    bev_backbone=None,
    occ_head=dict(
        _delete_=True,
        type='GaussianOccHead',
        in_channels=256,
        feat_channels=64,
        semantic_classes=18,
        voxels_size=[24, 300, 300],
        real_size=[-40.0, 40.0, -40.0, 40.0, -1.0, 5.4],
        contracted_ratio=0.66667,
        image_size=(256, 704),
        supervise='self',
        semantic_loss_weight=0.02,
        disparity_smoothness=0.001,
        weight_entropy_last=0.1,
        weight_distortion=0.1,
        semantic_sample_ratio=0.25,
        last_free=True,
        min_depth=0.1,
        max_depth=80.0,
        stepsize=0.5,
        render_size=(128, 352)))

data = dict(train=dict(gsocc_supervise=True))
