# 用 det10m 的权重跑自有车数据。模型结构和训练配置一致，只改数据路径。
# 相机顺序必须和训练时相同，否则 lidar2img 会对错图像。
#
# 数据可以是两种摆法：
#   每个相机一个目录，外加一个 lidar 目录，同一帧文件名里的数字相同；
#   或者每一帧一个目录，里面放六张图和一帧点云。
# 标定里每路相机要有内参，以及 ego2cam / cam2ego / lidar2cam 之一。
# 点云文件仍是雷达坐标，脚本用 lidar2ego 变到自车（x 朝前、y 朝左）。
_base_ = ['./streampetr_convnext_pp_det10m_nus.py']

demo_data_root = '/home/qlsong/dataset/test_data4'
demo_calib = '/home/qlsong/dataset/calibration.yaml'
load_from = 'work_dirs/streampetr_convnext_pp_det10m_nus/latest.pth'

# 训练 infos 里 cams 的插入顺序。
demo_cameras = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]
