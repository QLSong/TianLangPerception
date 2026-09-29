# 训练与测试

两份配置在 `projects/configs/TianLangEyes/`：

| 配置 | 图像骨干 |
| --- | --- |
| `streampetr_convnext_pp_lss_nus.py` | DINOv3 ConvNeXt-Tiny，冻结 |
| `streampetr_r50_pp_lss_nus.py` | ImageNet ResNet-50，整网可训 |

默认 `task = 'both'`，1 卡，batch size 1，24 epoch，AdamW `2e-4`，FP16。日志和权重在 `work_dirs/<配置名>/`，每个 epoch 存一次，最多留 3 个。

`num_gpus` 写在配置里，并且要和启动命令的 GPU 数一致。它只用来算每个 epoch 的迭代数，不会自动改学习率。

## 训练

在仓库根目录执行。`dist_train.sh` 会把根目录加进 `PYTHONPATH`。

```bash
bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py 1

bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_r50_pp_lss_nus.py 1
```

换卡数时同时改两处：启动命令的 GPU 数，以及配置里的 `num_gpus`。例如 8 卡：

```python
num_gpus = 8
batch_size = 1
```

```bash
bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py 8
```

指定目录或接着训：

```bash
bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py 1 \
    --work-dir work_dirs/convnext_both \
    --resume-from work_dirs/convnext_both/latest.pth
```

## 任务开关

配置顶部的 `task` 取 `det`、`occ` 或 `both`。没打开的分支在 `train()` 里被关掉梯度。

| `task` | 读点云 | 读 `gts/` | 更新的部分 |
| --- | --- | --- | --- |
| `det` | 是 | 否 | 图像骨干（ConvNeXt 除外）、PointPillars、检测头 |
| `occ` | 否 | 是 | LSS、UNet、占用头，以及未冻结的图像骨干 |
| `both` | 是 | 是 | 上面两边 |

ConvNeXt 配置里 `frozen=True`，无论 `task` 取什么，骨干都不更新。ResNet-50 的 BN 是可训练的（`frozen_stages=-1`，`norm_eval=False`）。

占用只吃当前帧的图像特征。检测按序列训练：`seq_mode=True`，`seq_split_num=2`，`queue_length=1`。

`task == 'occ'` 时，检测评测的 interval 被放到总迭代数之后，训练过程中不会跑 nuScenes bbox 评测。

## 几何

不要把两套范围改成同一套。检测框跟官方 StreamPETR，激光雷达和占用跟 Orin 上的图。

| | 数值 |
| --- | --- |
| 检测 `point_cloud_range` | `[-51.2, -51.2, -5, 51.2, 51.2, 3]` |
| 激光雷达 `lidar_pc_range` | `[-40, -40, -3, 40, 40, 1]`，voxel `0.2 m` |
| PointPillars 画布 | `400×400`，最多 12000 pillar，每个 32 点，输出 384 维 |
| LSS | `x/y [-40, 40]` 间隔 `0.4 m`，深度 `[1, 45)` 间隔 `0.5 m`，`z` 压成一层 |
| 图像 | `360×640`，缩放固定 0.4，不做旋转和翻转 |
| 占用头 | 18 类，`Dz=16`。损失与 MambaOcc `BEVOCCHead2D_V2` 相同：类别平衡 focal 再乘 100，外加 semantic scale、geometric scale（空类 17）和 Lovasz-softmax |

检测 query 384 个，时序 memory 1024，向后传播 128 个，decoder 3 层。每一层在图像交叉注意力之后，再用 `PETRMultiheadAttention` 看一次激光雷达 BEV。

## 测试

检测指标是 nuScenes 的 NDS / mAP。`task` 需要包含 `det`，否则结果里没有 `pts_bbox`。

```bash
bash tools/dist_test.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py \
    work_dirs/streampetr_convnext_pp_lss_nus/latest.pth \
    1 --eval bbox
```

占用没有单独的评测入口。训练日志里看 `loss_occ`、`loss_voxel_sem_scal`、`loss_voxel_geo_scal` 和 `loss_voxel_lovasz`。`task` 含 `occ` 时，`simple_test` 会在每个样本上附带 `occ`，是 softmax 之后的类别下标，`uint8`。测试 pipeline 不读 `gts/`，这条命令不会算占用损失。

只导出检测 json、不算指标：

```bash
bash tools/dist_test.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py \
    work_dirs/streampetr_convnext_pp_lss_nus/latest.pth \
    1 --format-only
```
