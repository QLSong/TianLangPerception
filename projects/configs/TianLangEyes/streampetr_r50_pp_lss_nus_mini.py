_base_ = ['./streampetr_r50_pp_lss_nus.py']

# 训练集里的第一个完整 scene（scene-0001，40 帧），用来把训练代码跑通。
# 采样器要求序列组数不少于 batch。这一个 scene 要切成至少 batch_size 组。
data = dict(
    train=dict(
        ann_file='./data/nuscenes/nuscenes2d_temporal_infos_train_mini.pkl',
        seq_split_num=4),
)
runner = dict(type='IterBasedRunner', max_iters=2000)
evaluation = dict(interval=100000)
checkpoint_config = dict(interval=500)
