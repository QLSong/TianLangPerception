# 四路环视带畸变进模型，IPM 用内外参把特征投到 BEV。权重和 det10m 训练配置一致。
_base_ = ['./streampetr_convnext_pp_det10m_nus.py']

demo_data_root = '/mnt/data/nas/datasets/test_my_data'
demo_calib = '/mnt/data/nas/datasets/test_my_data/calibration.yaml'
load_from = 'work_dirs/streampetr_convnext_pp_det10m_nus/latest.pth'

# 高×宽。1920×1536 正好缩到 1/3，不裁切。IPM 的 input_size 必须和这张图一致。
_input_hw = (512, 640)
ida_aug_conf = dict(final_dim=_input_hw)
model = dict(img_view_transformer=dict(input_size=_input_hw))

# 由光轴定的顺序：前、左、右、后。目录里没有 left/right 两路鱼眼。
demo_cameras = ['cam3', 'cam2', 'cam0', 'cam1']
demo_lidars = ['front_up', 'back_down']
