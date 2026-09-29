# 数据与预训练权重

检测和占用都读 `./data/nuscenes`。只训检测时可以不准备占用标签。`task` 含 `occ` 时，缺 `labels.npz` 会在取 batch 时直接报错。

## nuScenes

从 [nuScenes](https://www.nuscenes.org/download) 下载 trainval（以及要用 test 时的 test），解压到 `./data/nuscenes`，使 `samples/`、`sweeps/`、`maps/`、`v1.0-trainval/` 都在这一层。

生成带 2D 框和时序字段的 info。pkl 写在 `--root-path` 下，`--out-dir` 没有被脚本使用。

```bash
python tools/create_data_nusc.py \
    --root-path ./data/nuscenes \
    --extra-tag nuscenes2d \
    --version v1.0
```

`--version v1.0` 会先写 trainval，再写 test：

- `nuscenes2d_temporal_infos_train.pkl`
- `nuscenes2d_temporal_infos_val.pkl`
- `nuscenes2d_temporal_infos_test.pkl`

没有下载 `v1.0-test` 时，第二步会失败。前两个 pkl 已经写好，当前配置的 train / val 只用它们。mini 用 `--version v1.0-mini`，只生成 train 和 val。

train pkl 的样本数是 28130。配置里的 `num_iters_per_epoch = 28130 // (num_gpus * batch_size)` 按这个数算。

## Occ3D

占用标签用 [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D)。解压后每个 sample 一个 `labels.npz`，里面要有 `semantics` 和 `mask_camera`。

```text
data/nuscenes/gts/<scene-name>/<sample_token>/labels.npz
```

配置里的 `occ_root` 是 `./data/nuscenes/gts/`。info 里带 `scene_name` 时按上面的路径读；没有 scene 名时会扫一遍 `gts/`，用 sample token 做索引。

## 目录

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── gts/
│   └── scene-xxxx/<sample_token>/labels.npz
├── nuscenes2d_temporal_infos_train.pkl
└── nuscenes2d_temporal_infos_val.pkl
```

## 预训练权重

ResNet-50 配置使用 `torchvision://resnet50`，第一次建模型时自动下载。

ConvNeXt-Tiny 配置把骨干冻结，权重路径写死为：

```text
ckpts/dinov3_convnext_tiny_lvd1689m.safetensors
```

从 ModelScope 的 `facebook/dinov3-convnext-tiny-pretrain-lvd1689m` 下载 `model.safetensors`，放到这个路径。官方 `.pth` 也可以，加载时会把 DINOv3 / Hugging Face 的参数名映射过来。文件不存在会在建模型时直接报错。
