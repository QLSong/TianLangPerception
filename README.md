# TianLangEyes

nuScenes 上的相机 + 激光雷达多任务模型。检测沿用 [StreamPETR](https://arxiv.org/abs/2303.11926) 的时序 query，并在每一层图像交叉注意力之后再做一次 PointPillars BEV 交叉注意力。占用走相机 LSS，不吃点云。

计算图对齐 Orin 上的 TensorRT 导出：`PETRMultiheadAttention`（不用 flash attention）、关闭 checkpoint 和 grid mask、占用使用 `bev_pool_v2`。

## 结构

```text
6 路相机 ── ResNet-50 或 DINOv3 ConvNeXt-Tiny ── CPFPN (256)
                                                      │
                      ┌───────────────────────────────┴───────────────────────────────┐
                      │                                                               │
                      ▼                                                               ▼
              StreamPETR query                                                LSS + BEVPoolv2
                      │                                                               │
                      │ 每层 decoder 后再交叉注意                                       ▼
                      │                                                               UNet
LiDAR ── PointPillars (400×400, 384-d) ──┘                                             │
                      │                                                               ▼
                      ▼                                                    BEVOCCHead2D (18 类, Dz=16)
              10 类 3D 框
```

检测和占用可以分开训。配置里的 `task` 取 `det`、`occ` 或 `both`。没打开的分支参数会被冻结：只训检测时 LSS / UNet / 占用头不更新，只训占用时 PointPillars 和检测头不更新。

两套几何范围故意不一致，不要改成同一套：

| | 范围 | 用途 |
| --- | --- | --- |
| 检测框 | `[-51.2, -51.2, -5, 51.2, 51.2, 3]` | 与官方 StreamPETR 一致 |
| 激光雷达 / 占用 | `[-40, -40]` 到 `[40, 40]`，pillar `0.2 m`，LSS `0.4 m` | 与 Orin 图一致 |

图像输入 `360×640`。激光雷达 pillar 为 `(P, N, 9)`，画布 `400×400`，最多 `12000` 个 pillar、每个 `32` 个点。

## 配置

| 配置 | 图像骨干 | 权重 |
| --- | --- | --- |
| `projects/configs/TianLangEyes/streampetr_r50_pp_lss_nus.py` | ImageNet ResNet-50，整网可训 | `torchvision://resnet50` |
| `projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py` | DINOv3 ConvNeXt-Tiny，骨干冻结 | `ckpts/dinov3_convnext_tiny_lvd1689m.safetensors` |

`num_gpus` 和 `batch_size` 写在配置里，用来算每个 epoch 的迭代数。启动时的 GPU 数要和 `num_gpus` 一致。当前默认是 1 卡、batch size 1、24 epoch。

上游纯视觉 StreamPETR 配置仍在 `projects/configs/StreamPETR/`。

## 环境

安装步骤见 [docs/setup.md](docs/setup.md)。本仓库额外依赖：

- `mmsegmentation`，占用分支用它的 UNet

训练时的 BEV pooling 用 PyTorch `index_add`，不用编译 CUDA 扩展。

ConvNeXt 配置不使用 flash attention。

## 数据

nuScenes 放到 `./data/nuscenes`。占用标签用 [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) 的 `labels.npz`，目录为 `./data/nuscenes/gts/<scene>/<sample_token>/labels.npz`。

时序 info 由下面这条命令生成，得到 `nuscenes2d_temporal_infos_{train,val}.pkl`：

```bash
python tools/create_data_nusc.py \
    --root-path ./data/nuscenes \
    --extra-tag nuscenes2d \
    --version v1.0
```

只训检测时可以不准备 `gts/`。`task` 含 `occ` 时，缺标签会直接报错。

## 训练与测试

```bash
# ConvNeXt-Tiny + PointPillars + LSS，检测和占用一起训
bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py 1

# ResNet-50
bash tools/dist_train.sh \
    projects/configs/TianLangEyes/streampetr_r50_pp_lss_nus.py 1
```

只训其中一个任务时，把对应配置里的 `task` 改成 `det` 或 `occ`。

```bash
bash tools/dist_test.sh \
    projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py \
    work_dirs/streampetr_convnext_pp_lss_nus/latest.pth \
    1 --eval bbox
```

## 引用

检测头来自 StreamPETR：

```bibtex
@inproceedings{wang2023streampetr,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  booktitle={ICCV},
  year={2023}
}
```
