# 检测、数据、优化器和图像骨干与 streampetr_convnext_pp_lss_nus.py 相同。
# 占用换成 GaussianOcc：当前帧图像特征抬到收缩坐标体素，3D CNN 后采样成 Occ3D。
# 体素 24×300×300、范围 ±40 m、18 类和官方 nusc-sem-gs 一致。
# 图像仍是现有的 256×704 和 CPFPN，不改成官方的 384×640 ResNet。
# 3D Gaussian 光栅化扩展未编译，占用用 Occ3D 标签监督，不走光度渲染。
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
        image_size=(256, 704)))
