"""把多相机特征投到 BEV。

深度和上下文一起由 1×1 卷积预测，再用相机内参的逆把视锥变到自车坐标。
训练时 bev_pool_v2 是可导的 index_add；导出 ONNX 时符号换成 mmdeploy::bev_pool_v2。
"""
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS

from projects.mmdet3d_plugin.ops import bev_pool_v2


@NECKS.register_module(force=True)
class LSSViewTransformer(BaseModule):
    """Lift-Splat 视角变换。collapse_z 为真时把高度压成一层 BEV。"""

    def __init__(self, grid_config, input_size, downsample=16, in_channels=512,
                 out_channels=64, accelerate=False, sid=False, collapse_z=True):
        super().__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum(grid_config['depth'], input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(in_channels, self.D + self.out_channels, kernel_size=1)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]])

    def create_frustum(self, depth_cfg, input_size, downsample):
        h_in, w_in = input_size
        h_feat, w_feat = h_in // downsample, w_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float).view(-1, 1, 1).expand(-1, h_feat, w_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D - 1) *
                              torch.log((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, h_feat, w_feat)
        x = torch.linspace(0, w_in - 1, w_feat, dtype=torch.float).view(1, 1, w_feat).expand(self.D, h_feat, w_feat)
        y = torch.linspace(0, h_in - 1, h_feat, dtype=torch.float).view(1, h_feat, 1).expand(self.D, h_feat, w_feat)
        return torch.stack((x, y, d), -1)

    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        del ego2global
        b, n, _, _ = sensor2ego.shape
        points = self.frustum.to(sensor2ego) - post_trans.view(b, n, 1, 1, 1, 3)
        points = post_rots.transpose(-1, -2).view(b, n, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        fx, fy = cam2imgs[..., 0, 0], cam2imgs[..., 1, 1]
        cx, cy = cam2imgs[..., 0, 2], cam2imgs[..., 1, 2]
        k_inv = torch.zeros_like(cam2imgs)
        k_inv[..., 0, 0] = 1.0 / fx
        k_inv[..., 1, 1] = 1.0 / fy
        k_inv[..., 0, 2] = -cx / fx
        k_inv[..., 1, 2] = -cy / fy
        k_inv[..., 2, 2] = 1.0
        combine = sensor2ego[:, :, :3, :3].matmul(k_inv).view(b, n, 1, 1, 1, 3, 3)
        post_inv = post_rots.transpose(-1, -2).view(b, n, 1, 1, 1, 3, 3)
        combine = combine.matmul(post_inv)
        points = combine.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points = points + sensor2ego[:, :, :3, 3].view(b, n, 1, 1, 1, 3)
        points = bda.view(b, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def voxel_pooling_prepare_v2(self, coor):
        b, n, d, h, w, _ = coor.shape
        num_points = b * n * d * h * w
        ranks_depth = torch.arange(num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(num_points // d, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(b, n, 1, h, w).expand(b, n, d, h, w).flatten()
        coor = ((coor - self.grid_lower_bound.to(coor)) / self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(b, device=coor.device).reshape(b, 1).expand(b, num_points // b).reshape(num_points, 1)
        coor = torch.cat((coor, batch_idx), 1)
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if kept.sum() == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = coor[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = coor[:, 3] * (self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev = ranks_bev + coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev = ranks_bev + coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = ranks_bev[order], ranks_depth[order], ranks_feat[order]
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return (ranks_bev.int().contiguous(), ranks_depth.int().contiguous(),
                ranks_feat.int().contiguous(), interval_starts.contiguous(),
                interval_lengths.contiguous())

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks = self.voxel_pooling_prepare_v2(coor)
        if ranks[2] is None:
            dummy = feat.new_zeros(
                feat.shape[0], feat.shape[2], int(self.grid_size[2]),
                int(self.grid_size[1]), int(self.grid_size[0]))
            return torch.cat(dummy.unbind(dim=2), 1)
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = ranks
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]), int(self.grid_size[1]),
                          int(self.grid_size[0]), feat.shape[-1])
        bev_feat = bev_pool_v2(
            depth, feat, ranks_depth, ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts, interval_lengths)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def forward(self, inputs):
        x = inputs[0]
        b, n, c, h, w = x.shape
        x = self.depth_net(x.reshape(b * n, c, h, w))
        depth = x[:, :self.D].softmax(dim=1)
        tran = x[:, self.D:self.D + self.out_channels]
        return self.voxel_pooling_v2(
            self.get_ego_coor(*inputs[1:7]),
            depth.view(b, n, self.D, h, w),
            tran.view(b, n, self.out_channels, h, w)), depth
