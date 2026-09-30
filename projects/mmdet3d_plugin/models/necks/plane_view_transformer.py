"""按固定高度把每个相机的特征图投成 BEV，再用多尺度可变形注意力合成一张。"""
import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS

from projects.mmdet3d_plugin.models.utils.deform_cross_attn import ms_deform_attn


def ego_to_image(ego, sensor2ego, cam2imgs, post_rots, post_trans, bda):
    """自车坐标点投回图像像素。ego 是 (B, N, P, 3)，返回 u、v 和相机坐标系深度。"""
    xyz = torch.linalg.inv(bda)[:, None, None].matmul(ego.unsqueeze(-1)).squeeze(-1)
    rotation = sensor2ego[:, :, :3, :3]
    translation = sensor2ego[:, :, :3, 3]
    xyz = xyz - translation[:, :, None, :]
    cam = rotation.transpose(-1, -2)[:, :, None].matmul(xyz.unsqueeze(-1)).squeeze(-1)
    lifted = post_rots[:, :, None].matmul(cam2imgs[:, :, None].matmul(cam.unsqueeze(-1))).squeeze(-1)
    depth = lifted[..., 2]
    safe = depth.clamp(min=1e-5)
    view = torch.stack((lifted[..., 0] / safe, lifted[..., 1] / safe, depth), dim=-1)
    pix = post_rots[:, :, None].matmul(view.unsqueeze(-1)).squeeze(-1) + post_trans[:, :, None, :]
    return pix[..., 0], pix[..., 1], cam[..., 2]


def sample_plane(feat, sensor2ego, cam2imgs, post_rots, post_trans, bda, z, gx, gy, input_size):
    """把每个相机的特征图投到高度 z 的平面上，可见相机取平均。

    feat 是 (B, N, C, H, W)。gx、gy 是 BEV 格子中心，形状 (Y, X)。
    返回这一层的 BEV (B, C, Y, X) 和可见掩码 (B, Y, X)。
    """
    b, n, c, h, w = feat.shape
    ny, nx = gx.shape
    ego = torch.stack((gx, gy, torch.full_like(gx, float(z))), dim=-1)
    ego = ego.reshape(1, 1, ny * nx, 3).expand(b, n, -1, -1)
    u, v, depth = ego_to_image(ego, sensor2ego, cam2imgs, post_rots, post_trans, bda)
    h_in, w_in = input_size
    x_norm = u / (w_in - 1) * 2 - 1
    y_norm = v / (h_in - 1) * 2 - 1
    valid = (depth > 1e-5) & (x_norm > -1) & (x_norm < 1) & (y_norm > -1) & (y_norm < 1)
    grid = torch.stack((x_norm, y_norm), dim=-1).reshape(b * n, ny, nx, 2)
    sampled = torch.nn.functional.grid_sample(
        feat.reshape(b * n, c, h, w), grid, mode='bilinear',
        padding_mode='zeros', align_corners=True)
    sampled = sampled.view(b, n, c, ny, nx)
    visible = valid.view(b, n, 1, ny, nx).to(sampled.dtype)
    bev = (sampled * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1)
    return bev, valid.any(dim=1).view(b, ny, nx)


def aggregate_planes(attn, bev, valid):
    """把多层平面 BEV 聚成一张。bev 是 (B, C, Z, Y, X)，valid 是 (B, Z, Y, X)。"""
    b, c, z, y, x = bev.shape
    value = bev.permute(0, 2, 3, 4, 1).reshape(b, z * y * x, c)
    weight = valid.to(dtype=bev.dtype).unsqueeze(1)
    query = (bev * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1)
    query = query.permute(0, 2, 3, 1).reshape(b, y * x, c)
    gy, gx = torch.meshgrid(
        torch.linspace(0.5 / y, 1.0 - 0.5 / y, y, device=bev.device, dtype=bev.dtype),
        torch.linspace(0.5 / x, 1.0 - 0.5 / x, x, device=bev.device, dtype=bev.dtype),
        indexing='ij')
    ref = torch.stack((gx, gy), dim=-1).reshape(1, y * x, 1, 2).expand(b, -1, z, -1)
    spatial = bev.new_tensor([[y, x]] * z).to(dtype=torch.long)
    level_start = torch.arange(z, device=bev.device, dtype=torch.long) * (y * x)
    out = ms_deform_attn(
        attn, query, value, query.new_zeros(query.shape), None, ref,
        spatial, level_start, valid.permute(0, 2, 3, 1).reshape(b, y * x, z))
    return out.reshape(b, y, x, c).permute(0, 3, 1, 2).contiguous()


@NECKS.register_module(force=True)
class PlaneViewTransformer(BaseModule):
    """每个相机一张特征图，按内外参投到若干高度平面，再聚成一张 BEV。"""

    def __init__(self, grid_config, input_size, in_channels, out_channels=64,
                 bev_planes=(-1.0, 0.0, 1.0, 2.0), num_heads=4, num_points=4,
                 downsample=16):
        super().__init__()
        del downsample
        planes = tuple(float(z) for z in bev_planes)
        if len(planes) < 2:
            raise ValueError('至少需要两个高度平面')
        if out_channels % num_heads != 0:
            raise ValueError('out_channels 必须能被 num_heads 整除')
        self.grid_config = grid_config
        self.input_size = tuple(input_size)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.bev_planes = planes
        x, y = grid_config['x'], grid_config['y']
        self.grid_lower_bound = torch.Tensor([x[0], y[0]])
        self.grid_interval = torch.Tensor([x[2], y[2]])
        self.grid_size = torch.Tensor([(x[1] - x[0]) / x[2], (y[1] - y[0]) / y[2]])
        self.feat_net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.z_attn = MultiScaleDeformableAttention(
            embed_dims=out_channels, num_heads=num_heads, num_levels=len(planes),
            num_points=num_points, dropout=0.0, batch_first=True)

    def bev_centers(self, device, dtype):
        nx = int(self.grid_size[0].item())
        ny = int(self.grid_size[1].item())
        interval = self.grid_interval.to(device=device, dtype=dtype)
        lower = self.grid_lower_bound.to(device=device, dtype=dtype)
        xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * interval[0] + lower[0]
        ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * interval[1] + lower[1]
        return torch.meshgrid(ys, xs, indexing='ij')[::-1]

    def forward(self, inputs):
        img, sensor2ego, _, cam2imgs, post_rots, post_trans, bda = inputs
        b, n, c, h, w = img.shape
        feat = self.feat_net(img.reshape(b * n, c, h, w)).view(b, n, self.out_channels, h, w)
        gx, gy = self.bev_centers(feat.device, feat.dtype)
        planes, valids = [], []
        for z in self.bev_planes:
            bev, valid = sample_plane(
                feat, sensor2ego, cam2imgs, post_rots, post_trans, bda,
                z, gx, gy, self.input_size)
            planes.append(bev)
            valids.append(valid)
        bev = aggregate_planes(self.z_attn, torch.stack(planes, dim=2), torch.stack(valids, dim=1))
        return bev, None
