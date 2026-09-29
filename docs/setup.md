# 环境

仓库里已经带了 `mmdetection3d` v1.0.0rc6，直接在本目录安装，不要再 clone 一份。TianLangEyes 的配置使用 `PETRMultiheadAttention`，不需要 flash attention。

配套版本：

| 包 | 版本 |
| --- | --- |
| Python | 3.8 |
| PyTorch | 1.9.0+cu111 |
| mmcv-full | 1.6.0 |
| mmdet | 2.28.2 |
| mmsegmentation | 0.30.0 |
| mmdet3d | 1.0.0rc6（仓库内） |

占用分支用 mmsegmentation 的 UNet。读 DINOv3 的 `.safetensors` 需要 `safetensors`。`nuscenes-devkit` 随 mmdet3d 装上，生成 info 和检测评测都会用到。

## 安装

```bash
conda create -n tianlang python=3.8 -y
conda activate tianlang

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.6.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2 mmsegmentation==0.30.0 safetensors

cd /path/to/TianLangEyes
pip install -e mmdetection3d
```

`mmdetection3d` 的运行依赖把 `numpy` 限制在 1.24 以下。驱动需要能跑 CUDA 11.1 的 PyTorch wheel。

训练时的 BEV pooling 是 PyTorch 的 `index_add`，不用编译 CUDA 扩展。导出 ONNX 时符号仍然是 `mmdeploy::bev_pool_v2`，给 Orin 上的插件用。

## 检查

```bash
python -c "import torch, mmcv, mmdet, mmseg, mmdet3d; print(torch.__version__, mmdet3d.__version__)"
```

应看到 `1.9.0+cu111` 和 `1.0.0rc6`。
