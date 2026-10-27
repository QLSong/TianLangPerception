"""按固定高度把每个相机的特征图投成 BEV，再用卷积把多层合成一张。"""
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS


def rational_distort(x, y, dist):
    """OpenCV rational 模型。dist 是 (B, N, 8)：k1, k2, p1, p2, k3, k4, k5, k6。

    x、y 是相机归一化坐标。畸变加在乘内参之前，图像本身保持带畸变的像素。
    """
    k1, k2, p1, p2, k3, k4, k5, k6 = [item[..., None] for item in dist.unbind(-1)]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    scale = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6).clamp(min=1e-8)
    xd = x * scale + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * scale + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return xd, yd


def ego_to_image(ego, sensor2ego, cam2imgs, post_rots, post_trans, bda, distortion=None):
    """自车坐标点投回图像像素。ego 是 (B, N, P, 3)，返回 u、v 和相机坐标系深度。

    distortion 是 (B, N, 8) 时，先在归一化平面做 rational 畸变，再用内参。
    不传时和原来的针孔投影一致。
    """
    xyz = torch.linalg.inv(bda)[:, None, None].matmul(ego.unsqueeze(-1)).squeeze(-1)
    rotation = sensor2ego[:, :, :3, :3]
    translation = sensor2ego[:, :, :3, 3]
    xyz = xyz - translation[:, :, None, :]
    cam = rotation.transpose(-1, -2)[:, :, None].matmul(xyz.unsqueeze(-1)).squeeze(-1)
    if distortion is not None:
        depth = cam[..., 2]
        safe = depth.clamp(min=1e-5)
        xn, yn = rational_distort(cam[..., 0] / safe, cam[..., 1] / safe, distortion)
        cam = torch.stack((xn * safe, yn * safe, depth), dim=-1)
    lifted = post_rots[:, :, None].matmul(cam2imgs[:, :, None].matmul(cam.unsqueeze(-1))).squeeze(-1)
    depth = lifted[..., 2]
    safe = depth.clamp(min=1e-5)
    view = torch.stack((lifted[..., 0] / safe, lifted[..., 1] / safe, depth), dim=-1)
    pix = post_rots[:, :, None].matmul(view.unsqueeze(-1)).squeeze(-1) + post_trans[:, :, None, :]
    return pix[..., 0], pix[..., 1], cam[..., 2]


def sample_planes(feat, sensor2ego, cam2imgs, post_rots, post_trans, bda, zs, gx, gy, input_size,
                  distortion=None):
    """把每个相机的特征图一次投到全部高度平面上，可见相机取平均。

    feat 是 (B, N, C, H, W)。zs 是 (Z,)，gx、gy 是 (Y, X)。
    返回 (B, C, Z, Y, X) 和可见掩码 (B, Z, Y, X)。
    """
    b, n, c, h, w = feat.shape
    ny, nx = gx.shape
    z = zs.shape[0]
    ego = torch.stack((
        gx.unsqueeze(0).expand(z, -1, -1),
        gy.unsqueeze(0).expand(z, -1, -1),
        zs.view(z, 1, 1).expand(z, ny, nx),
    ), dim=-1)
    ego = ego.reshape(1, 1, z * ny * nx, 3).expand(b, n, -1, -1)
    u, v, depth = ego_to_image(
        ego, sensor2ego, cam2imgs, post_rots, post_trans, bda, distortion)
    h_in, w_in = input_size
    x_norm = u / (w_in - 1) * 2 - 1
    y_norm = v / (h_in - 1) * 2 - 1
    valid = (depth > 1e-5) & (x_norm > -1) & (x_norm < 1) & (y_norm > -1) & (y_norm < 1)
    grid = torch.stack((x_norm, y_norm), dim=-1).reshape(b * n, z * ny, nx, 2).contiguous()
    # 特征图只有十几像素，BEV 有几百像素。cuDNN 的 grid_sample 不支持这种输出，会报 NOT_SUPPORTED。
    with torch.backends.cudnn.flags(enabled=False):
        sampled = torch.nn.functional.grid_sample(
            feat.reshape(b * n, c, h, w).contiguous(), grid, mode='bilinear',
            padding_mode='zeros', align_corners=True)
    sampled = sampled.view(b, n, c, z, ny, nx)
    valid = valid.view(b, n, z, ny, nx)
    visible = valid.unsqueeze(2).to(sampled.dtype)
    bev = (sampled * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1)
    return bev, valid.any(dim=1)


def aggregate_planes(fuse, bev, valid):
    """把多层平面 BEV 沿通道拼起来，用卷积合成一张。

    bev 是 (B, C, Z, Y, X)，valid 是 (B, Z, Y, X)。看不见的格子先置零。
    """
    b, c, z, y, x = bev.shape
    weight = valid.to(dtype=bev.dtype)[:, None]
    stacked = (bev * weight).permute(0, 2, 1, 3, 4).reshape(b, z * c, y, x)
    return fuse(stacked)


@NECKS.register_module(force=True)
class PlaneViewTransformer(BaseModule):
    """每个相机一张特征图，按内外参投到若干高度平面，再用卷积聚成一张 BEV。"""

    def __init__(self, grid_config, input_size, in_channels, out_channels=64,
                 bev_planes=(-1.0, 0.0, 1.0, 2.0), downsample=16):
        super().__init__()
        del downsample
        planes = tuple(float(z) for z in bev_planes)
        if len(planes) < 2:
            raise ValueError('至少需要两个高度平面')
        self.grid_config = grid_config
        self.input_size = tuple(input_size)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.bev_planes = planes
        x, y = grid_config['x'], grid_config['y']
        nx = int(round((x[1] - x[0]) / x[2]))
        ny = int(round((y[1] - y[0]) / y[2]))
        xs = (torch.arange(nx) + 0.5) * x[2] + x[0]
        ys = (torch.arange(ny) + 0.5) * y[2] + y[0]
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('gx', gx, persistent=False)
        self.register_buffer('gy', gy, persistent=False)
        self.register_buffer('plane_z', torch.tensor(planes), persistent=False)
        self.feat_net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.z_fuse = nn.Sequential(
            nn.Conv2d(out_channels * len(planes), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        img, sensor2ego, _, cam2imgs, post_rots, post_trans, bda = inputs[:7]
        distortion = inputs[7] if len(inputs) > 7 else None
        b, n, c, h, w = img.shape
        feat = self.feat_net(img.reshape(b * n, c, h, w)).view(b, n, self.out_channels, h, w)
        bev, valid = sample_planes(
            feat, sensor2ego, cam2imgs, post_rots, post_trans, bda,
            self.plane_z.to(dtype=feat.dtype), self.gx, self.gy, self.input_size,
            distortion)
        return aggregate_planes(self.z_fuse, bev, valid), None
