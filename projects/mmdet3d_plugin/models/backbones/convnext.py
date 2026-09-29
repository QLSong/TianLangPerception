"""DINOv3 ConvNeXt 骨干。

参数名按官方 .pth 和 Hugging Face / ModelScope 的 safetensors 做了映射。
当前配置用 tiny，out_indices 取 stage2 和 stage3，并冻结全部权重。
"""
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import BACKBONES

CONVNEXT_ARCH = {
    'tiny': dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)),
    'small': dict(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768)),
    'base': dict(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)),
    'large': dict(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536)),
}
DINOV3_CKPT = {
    'tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
    'small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
    'base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
    'large': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
}
logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """ConvNeXt 的 LayerNorm。下采样层用 channels_first，block 里用 channels_last。"""

    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(normalized_shape))
        self.bias = nn.Parameter(torch.empty(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = self.gamma * x
        return shortcut + x.permute(0, 3, 1, 2)


def _remap_key(name):
    """把官方权重和 Hugging Face 权重的参数名映射到本模块。对不上的 head、norm 直接丢掉。"""
    import re
    if name.startswith('module.'):
        name = name[7:]
    if name.startswith('backbone.'):
        name = name[len('backbone.'):]
    if name.startswith('model.'):
        name = name[len('model.'):]
    if name.startswith(('norms.', 'projectors.', 'head.', 'pool.')):
        return None
    down = re.match(r'stages\.(\d+)\.downsample_layers\.(\d+)\.(.+)', name)
    if down:
        return 'downsample_layers.{}.{}.{}'.format(down.group(1), down.group(2), down.group(3))
    layer = re.match(r'stages\.(\d+)\.layers\.(\d+)\.(.+)', name)
    if layer:
        rest = layer.group(3)
        rest = rest.replace('depthwise_conv', 'dwconv')
        rest = rest.replace('pointwise_conv', 'pwconv')
        rest = rest.replace('layer_norm', 'norm')
        return 'stages.{}.{}.{}'.format(layer.group(1), layer.group(2), rest)
    if name.startswith('layer_norm.'):
        return 'norm.' + name[len('layer_norm.'):]
    return name


def _read_checkpoint(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'model', 'teacher'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _load_dinov3(model, path):
    ckpt = _read_checkpoint(path)
    cleaned = {}
    for key, value in ckpt.items():
        name = _remap_key(key)
        if name is not None:
            cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    missing = [k for k in missing if not k.startswith(('norms.', 'norm.'))]
    if missing or unexpected:
        logger.warning('DINOv3 load missing=%s unexpected=%s', missing[:8], unexpected[:8])
    else:
        logger.info('loaded DINOv3 ConvNeXt from %s', path)


@BACKBONES.register_module()
class ConvNeXt(nn.Module):
    def __init__(self, arch='tiny', out_indices=(2, 3), pretrained=None, frozen=False, **kwargs):
        super().__init__()
        cfg = CONVNEXT_ARCH[arch]
        depths, dims = cfg['depths'], cfg['dims']
        self.out_indices = out_indices
        self.frozen = frozen
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        self.stages = nn.ModuleList([
            nn.Sequential(*[Block(dims[i]) for _ in range(depths[i])])
            for i in range(4)
        ])
        # 最后一层 LayerNorm 只为装下完整 DINOv3 权重，前向不用它。
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        if pretrained:
            if not os.path.isfile(pretrained):
                raise FileNotFoundError(
                    f'DINOv3 ConvNeXt checkpoint not found: {pretrained}. '
                    'Hugging Face blocks this file in this region. '
                    'Download model.safetensors from ModelScope '
                    f'facebook/dinov3-convnext-{arch}-pretrain-lvd1689m '
                    'and place it at this path. Official .pth files are also accepted.')
            _load_dinov3(self, pretrained)
        if self.frozen:
            self._freeze()

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super().train(False if self.frozen else mode)
        if self.frozen:
            self._freeze()
        return self

    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                return self._forward_features(x)
        return self._forward_features(x)

    def _forward_features(self, x):
        outs = []
        for i, (down, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = stage(down(x))
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        # 权重在 __init__ 里已经加载，避免 mmcv 再按默认方式初始化一遍。
        if self.frozen:
            self._freeze()
        return
