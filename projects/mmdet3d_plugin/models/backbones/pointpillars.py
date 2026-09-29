"""PointPillars 骨干，输出给检测头做交叉注意力。

输入 pillar 是 (B, 9, P, N)，坐标是 (B, P, 2)，顺序 (y, x)。
PillarFeatureNet 压成点特征，Scatter 铺到 ny×nx 画布，再过 SECOND 和 SECONDFPN。
输出通道数是 384，对应配置里的 lidar_in_channels。
"""
from typing import List, Sequence

import torch
import torch.nn as nn
from mmdet.models import BACKBONES


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=9, feat_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels, 1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pillars):
        return self.net(pillars).amax(dim=3)


class PointPillarsScatter(nn.Module):
    def __init__(self, ny, nx, channels=64):
        super().__init__()
        self.ny = ny
        self.nx = nx
        self.channels = channels

    def forward(self, feat, coords):
        b, c, p = feat.shape
        canvas = feat.new_zeros(b, c, self.ny * self.nx)
        y = coords[..., 0].clamp(0, self.ny - 1)
        x = coords[..., 1].clamp(0, self.nx - 1)
        idx = (y * self.nx + x).to(dtype=torch.int64)
        idx = idx.unsqueeze(1).expand(b, c, p)
        canvas = canvas.scatter(2, idx, feat)
        return canvas.view(b, c, self.ny, self.nx)


class SECOND(nn.Module):
    def __init__(self, in_channels=64, layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2), out_channels=(64, 128, 256)):
        super().__init__()
        blocks = []
        cin = in_channels
        for num, stride, cout in zip(layer_nums, layer_strides, out_channels):
            layers: List[nn.Module] = [
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
            for _ in range(int(num) - 1):
                layers += [
                    nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                ]
            blocks.append(nn.Sequential(*layers))
            cin = cout
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class SECONDFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256),
                 out_channels=(128, 128, 128),
                 upsample_strides=(1, 2, 4)):
        super().__init__()
        deblocks = []
        for cin, cout, stride in zip(in_channels, out_channels, upsample_strides):
            if stride > 1:
                up = nn.ConvTranspose2d(cin, cout, stride, stride=stride, bias=False)
            else:
                up = nn.Conv2d(cin, cout, 1, bias=False)
            deblocks.append(nn.Sequential(
                up, nn.BatchNorm2d(cout), nn.ReLU(inplace=True)))
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, feats: Sequence[torch.Tensor]):
        return torch.cat([d(f) for d, f in zip(self.deblocks, feats)], dim=1)


@BACKBONES.register_module()
class PointPillars(nn.Module):
    def __init__(self, ny, nx, in_channels=9, feat_channels=64, **kwargs):
        super().__init__()
        self.voxel_encoder = PillarFeatureNet(in_channels, feat_channels)
        self.middle_encoder = PointPillarsScatter(ny, nx, feat_channels)
        self.backbone = SECOND()
        self.neck = SECONDFPN()

    def forward(self, pillars, coords):
        feat = self.voxel_encoder(pillars)
        bev = self.middle_encoder(feat, coords)
        return self.neck(self.backbone(bev))
