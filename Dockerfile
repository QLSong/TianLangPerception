# 本机 streampetr 环境：Python 3.8、CUDA 11.8、PyTorch 2.0.1、mmcv-full 1.7.2。
# 显卡架构按 RTX 3090 / 4090（8.6、8.9）编译 mmdet3d 的 CUDA 算子。
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        build-essential \
        git \
        ninja-build \
        ca-certificates \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade \
        pip==24.2 setuptools==75.1.0 wheel==0.44.0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.23.5 \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir mmcv-full==1.7.2 \
        -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

RUN pip install --no-cache-dir \
        mmdet==2.28.2 \
        mmcls==0.25.0 \
        mmsegmentation==0.30.0 \
        einops==0.8.1 \
        safetensors==0.5.3 \
        pyquaternion==0.9.9 \
        opencv-python==4.8.1.78 \
        Shapely==1.8.5.post1

# WORKDIR /opt/TianLangEyes
# COPY mmdetection3d /opt/TianLangEyes/mmdetection3d
# RUN pip install --no-cache-dir -e /opt/TianLangEyes/mmdetection3d

# COPY projects /opt/TianLangEyes/projects
# COPY tools /opt/TianLangEyes/tools

# ENV PYTHONPATH=/opt/TianLangEyes
# WORKDIR /opt/TianLangEyes
CMD ["bash"]
