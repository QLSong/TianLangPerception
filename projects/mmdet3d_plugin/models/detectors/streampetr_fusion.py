"""相机 + 激光雷达多任务检测器。

检测走 StreamPETR 时序 query，每一层图像交叉注意力后再看一次 PointPillars BEV。
占用把 LSS BEV 和同一份雷达 BEV 相加，再过 UNet 和占用头。
`use_lidar` 关掉时，两个任务都不再读点云。
`tasks` 取 det、occ 或 both；没打开的分支在 train() 里关掉梯度。
图像骨干可以是可训练的 ResNet-50，或冻结的 DINOv3 ConvNeXt。
交叉注意力用 PETRMultiheadAttention，占用 pooling 的导出符号是 bev_pool_v2。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models import build_backbone, build_head, build_neck
from mmseg.models.backbones.unet import UNet  # noqa: F401
from mmseg.models import build_backbone as build_mmseg_backbone

from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D


def parse_tasks(tasks):
    """把 'det'、'occ'、'both' 收成要训练的任务名。"""
    if isinstance(tasks, str):
        if tasks == 'both':
            return ('det', 'occ')
        if tasks in ('det', 'occ'):
            return (tasks,)
        raise ValueError("tasks must be 'det', 'occ', or 'both'")
    names = tuple(tasks)
    if not names or any(name not in ('det', 'occ') for name in names):
        raise ValueError("tasks must be 'det', 'occ', or 'both'")
    return names


def _keep_task_losses(losses, tasks):
    """单任务训练时丢掉另一个任务的 loss，避免被日志和总 loss 算进去。"""
    if 'det' in tasks and 'occ' in tasks:
        return losses
    det_marks = ('loss_cls', 'loss_bbox', 'loss_iou')
    occ_marks = ('loss_occ', 'loss_voxel_')
    kept = {}
    for key, value in losses.items():
        if 'det' not in tasks and any(mark in key for mark in det_marks):
            continue
        if 'occ' not in tasks and any(mark in key for mark in occ_marks):
            continue
        kept[key] = value
    return kept


def align_bev(lidar_bev, lidar_range, camera_bev, camera_range):
    """把雷达 BEV 采样到相机 BEV 的格子上。范围和尺寸一致时直接返回。

    两个 BEV 都是 (B, C, Y, X)，下标 0 是该轴的最小值。
    range 是 (x0, y0, x1, y1)。
    """
    if (lidar_bev.shape[-2:] == camera_bev.shape[-2:]
            and all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(lidar_range, camera_range))):
        return lidar_bev
    x0, y0, x1, y1 = [float(v) for v in camera_range]
    lx0, ly0, lx1, ly1 = [float(v) for v in lidar_range]
    height, width = camera_bev.shape[-2:]
    ys = torch.linspace(y0 + 0.5 * (y1 - y0) / height, y1 - 0.5 * (y1 - y0) / height, height, device=lidar_bev.device)
    xs = torch.linspace(x0 + 0.5 * (x1 - x0) / width, x1 - 0.5 * (x1 - x0) / width, width, device=lidar_bev.device)
    gy = ys.view(-1, 1).expand(height, width)
    gx = xs.view(1, -1).expand(height, width)
    grid_x = (gx - lx0) / (lx1 - lx0) * 2 - 1
    grid_y = (gy - ly0) / (ly1 - ly0) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(lidar_bev.shape[0], -1, -1, -1)
    sampled = F.grid_sample(
        lidar_bev.float(), grid.float(), mode='bilinear', padding_mode='zeros', align_corners=False)
    return sampled.to(dtype=lidar_bev.dtype)


def _set_requires_grad(module, enabled):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = enabled


def voxelize_pillars(points, voxel_size, pc_range, max_points, max_voxels):
    """把一帧点云收成 pillar，计算留在点所在的设备上。

    返回 (max_voxels, max_points, 9) 和 (max_voxels, 2)。
    9 维是 xyz、强度、相对 pillar 均值的 xyz、相对 pillar 中心的 xy。
    坐标顺序是 (y, x)。每个 pillar 保留点云里最先出现的 max_points 个点，
    pillar 本身也按首次出现顺序截到 max_voxels。
    """
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    pts = points.detach().float()
    device = pts.device
    x1, y1, z1, x2, y2, z2 = [float(v) for v in pc_range]
    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    nx = int(round((x2 - x1) / vx))
    ny = int(round((y2 - y1) / vy))
    pillars = pts.new_zeros((max_voxels, max_points, 9))
    coords = torch.zeros((max_voxels, 2), dtype=torch.int32, device=device)
    if pts.numel() == 0 or pts.shape[0] == 0:
        return pillars, coords
    if pts.shape[1] < 4:
        raise ValueError('points need at least x,y,z,intensity')
    keep = (
        (pts[:, 0] >= x1) & (pts[:, 0] < x2)
        & (pts[:, 1] >= y1) & (pts[:, 1] < y2)
        & (pts[:, 2] >= z1) & (pts[:, 2] < z2))
    pts = pts[keep]
    if pts.shape[0] == 0:
        return pillars, coords
    ix = torch.clamp(((pts[:, 0] - x1) / vx).floor(), 0, nx - 1).long()
    iy = torch.clamp(((pts[:, 1] - y1) / vy).floor(), 0, ny - 1).long()
    keys = iy * nx + ix
    order = torch.argsort(keys, stable=True)
    keys, pts, ix, iy = keys[order], pts[order], ix[order], iy[order]
    change = torch.ones(keys.shape[0], dtype=torch.bool, device=device)
    change[1:] = keys[1:] != keys[:-1]
    group = torch.cumsum(change, 0) - 1
    starts = torch.where(change)[0]
    rank = torch.arange(keys.shape[0], device=device) - starts[group]
    first_orig = order[starts]
    appear = torch.argsort(first_orig, stable=True)
    if appear.shape[0] > max_voxels:
        appear = appear[:max_voxels]
    n_voxels = int(appear.shape[0])
    slot_of_group = torch.full((starts.shape[0],), -1, dtype=torch.long, device=device)
    slot_of_group[appear] = torch.arange(n_voxels, device=device)
    valid = (rank < max_points) & (slot_of_group[group] >= 0)
    slot = slot_of_group[group][valid]
    rank = rank[valid]
    chosen = pts[valid]
    pillars[slot, rank, :4] = chosen[:, :4]
    coords[:n_voxels, 0] = iy[starts[appear]].to(torch.int32)
    coords[:n_voxels, 1] = ix[starts[appear]].to(torch.int32)
    ones = torch.ones(slot.shape[0], device=device, dtype=pts.dtype)
    counts = pts.new_zeros(n_voxels).scatter_add_(0, slot, ones)
    sum_xyz = pts.new_zeros(n_voxels, 3).scatter_add_(0, slot.unsqueeze(1).expand_as(chosen[:, :3]), chosen[:, :3])
    mean = sum_xyz / counts.clamp(min=1).unsqueeze(1)
    filled = torch.arange(max_points, device=device).unsqueeze(0) < counts.unsqueeze(1)
    xyz = pillars[:n_voxels, :, :3]
    pillars[:n_voxels, :, 4:7] = torch.where(filled.unsqueeze(-1), xyz - mean.unsqueeze(1), xyz.new_zeros(()))
    cx = (coords[:n_voxels, 1].to(pts.dtype) + 0.5) * vx + x1
    cy = (coords[:n_voxels, 0].to(pts.dtype) + 0.5) * vy + y1
    pillars[:n_voxels, :, 7] = torch.where(filled, xyz[:, :, 0] - cx.unsqueeze(1), xyz.new_zeros(()))
    pillars[:n_voxels, :, 8] = torch.where(filled, xyz[:, :, 1] - cy.unsqueeze(1), xyz.new_zeros(()))
    return pillars, coords


@DETECTORS.register_module()
class StreamPETRFusion(Petr3D):
    def __init__(self,
                 pts_backbone=None,
                 img_view_transformer=None,
                 bev_backbone=None,
                 occ_head=None,
                 lidar_pc_range=None,
                 lidar_voxel_size=None,
                 max_voxels=12000,
                 max_points=32,
                 tasks='both',
                 use_lidar=True,
                 lidar_bev_channels=384,
                 **kwargs):
        super().__init__(**kwargs)
        self.tasks = parse_tasks(tasks)
        self.use_lidar = bool(use_lidar)
        self.lidar_backbone = build_backbone(pts_backbone) if pts_backbone else None
        self.img_view_transformer = build_neck(img_view_transformer) if img_view_transformer else None
        self.bev_backbone = build_mmseg_backbone(bev_backbone) if bev_backbone else None
        bev_out = 64
        if bev_backbone is not None:
            bev_out = bev_backbone.get('base_channels', 64)
        occ_in = occ_head.get('in_dim', 256) if occ_head else 256
        self.bev_neck = nn.Conv2d(bev_out, occ_in, kernel_size=1) if self.bev_backbone else None
        self.occ_head = build_head(occ_head) if occ_head else None
        lss_channels = img_view_transformer.get('out_channels', 64) if img_view_transformer else 64
        self.lidar_occ_fuse = None
        if self.lidar_backbone is not None and self.img_view_transformer is not None:
            fuse = nn.Conv2d(lidar_bev_channels, lss_channels, kernel_size=1, bias=True)
            nn.init.zeros_(fuse.weight)
            nn.init.zeros_(fuse.bias)
            self.lidar_occ_fuse = fuse
        self.lidar_pc_range = lidar_pc_range
        self.lidar_voxel_size = lidar_voxel_size
        self.max_voxels = max_voxels
        self.max_points = max_points
        self._apply_task_freeze()

    def _apply_task_freeze(self):
        train_det = 'det' in self.tasks
        train_occ = 'occ' in self.tasks
        _set_requires_grad(self.lidar_backbone, self.use_lidar and (train_det or train_occ))
        _set_requires_grad(self.lidar_occ_fuse, self.use_lidar and train_occ)
        _set_requires_grad(self.pts_bbox_head, train_det)
        _set_requires_grad(self.img_view_transformer, train_occ)
        _set_requires_grad(self.bev_backbone, train_occ)
        _set_requires_grad(self.bev_neck, train_occ)
        _set_requires_grad(self.occ_head, train_occ)

    def train(self, mode=True):
        super().train(mode)
        self._apply_task_freeze()
        return self

    def _as_points(self, points):
        if points is None:
            return None
        if hasattr(points, 'data'):
            points = points.data
        if isinstance(points, (list, tuple)) and points and isinstance(points[0], (list, tuple)):
            points = points[0]
        return points

    def _lidar_xy_range(self):
        x0, y0, _, x1, y1, _ = [float(v) for v in self.lidar_pc_range]
        return (x0, y0, x1, y1)

    def _camera_xy_range(self):
        grid = self.img_view_transformer.grid_config
        return (grid['x'][0], grid['y'][0], grid['x'][1], grid['y'][1])

    def _prepare_lidar(self, points):
        if not self.use_lidar:
            return None
        lidar_bev = self.encode_lidar(points)
        if lidar_bev is None:
            raise RuntimeError('use_lidar is on but the batch has no points')
        return lidar_bev

    def encode_lidar(self, points):
        points = self._as_points(points)
        if points is None or self.lidar_backbone is None or not self.use_lidar:
            return None
        if not isinstance(points, (list, tuple)):
            points = [points]
        device = next(self.lidar_backbone.parameters()).device
        pillars, coords = [], []
        for pts in points:
            if hasattr(pts, 'tensor'):
                pts = pts.tensor
            p, c = voxelize_pillars(
                pts.to(device), self.lidar_voxel_size, self.lidar_pc_range,
                self.max_points, self.max_voxels)
            pillars.append(p)
            coords.append(c)
        pillars = torch.stack(pillars, 0).permute(0, 3, 1, 2).contiguous()
        coords = torch.stack(coords, 0)
        return self.lidar_backbone(pillars, coords)

    def _lss_inputs(self, img_feats, extrinsics, intrinsics):
        """外参是自车到相机。LSS 的 BEV 落在自车坐标，x 朝前、y 朝左。"""
        b, n = img_feats.shape[:2]
        device = img_feats.device
        if extrinsics.dim() == 5:
            extrinsics = extrinsics[:, -1]
            intrinsics = intrinsics[:, -1]
        sensor2ego = torch.linalg.inv(extrinsics)
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, n, 1, 1)
        cam2imgs = intrinsics[..., :3, :3]
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
        post_trans = torch.zeros(b, n, 3, device=device)
        bda = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
        return [img_feats, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda]

    def forward_occ(self, img_feats, data, lidar_bev=None):
        if self.img_view_transformer is None:
            return None
        bev, _ = self.img_view_transformer(
            self._lss_inputs(img_feats, data['extrinsics'], data['intrinsics']))
        if self.use_lidar:
            if lidar_bev is None or self.lidar_occ_fuse is None:
                raise RuntimeError('use_lidar is on but occupancy has no lidar BEV')
            aligned = align_bev(lidar_bev, self._lidar_xy_range(), bev, self._camera_xy_range())
            bev = bev + self.lidar_occ_fuse(aligned)
        dec = self.bev_backbone(bev)
        if isinstance(dec, (list, tuple)):
            dec = dec[-1]
        return self.occ_head(self.bev_neck(dec))

    def forward_train(self, img_metas=None, gt_bboxes_3d=None, gt_labels_3d=None,
                      gt_labels=None, gt_bboxes=None, gt_bboxes_ignore=None,
                      depths=None, centers2d=None, points=None,
                      voxel_semantics=None, mask_camera=None, **data):
        train_det = 'det' in self.tasks
        train_occ = 'occ' in self.tasks
        if train_det and self.test_flag:
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        t = data['img'].size(1)
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)
        # prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        # if t - self.num_frame_backbone_grads > 0:
        #     self.eval()
        #     with torch.no_grad():
        #         prev_img_feats = self.extract_feat(
        #             prev_img, t - self.num_frame_backbone_grads, True)
        #     self.train()
        #     data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        # else:
        data['img_feats'] = rec_img_feats
        losses = dict()
        lidar_bev = self._prepare_lidar(points) if (train_det or train_occ) else None
        if train_det:
            if lidar_bev is not None:
                data['lidar_bev'] = lidar_bev.unsqueeze(1).repeat(1, t, 1, 1, 1)
            losses.update(self.obtain_history_memory(
                gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, img_metas,
                centers2d, depths, gt_bboxes_ignore, **data))
        if train_occ:
            if voxel_semantics is None:
                raise RuntimeError('task includes occ but the batch has no voxel_semantics')
            occ = self.forward_occ(data['img_feats'][:, -1], data, lidar_bev)
            losses.update(self.occ_head.loss(occ, voxel_semantics, mask_camera))
        return _keep_task_losses(losses, self.tasks)

    def _points_for_test(self, points):
        """测试前向把点云收成 (B, N, C)，还原成 encode_lidar 要的逐帧列表。"""
        points = self._as_points(points)
        if torch.is_tensor(points) and points.dim() == 3:
            points = [points[i] for i in range(points.size(0))]
        return points

    def simple_test(self, img_metas, **data):
        data['img_feats'] = self.extract_img_feat(data['img'], 1)
        bbox_list = [dict() for _ in range(len(img_metas))]
        need_lidar = self.use_lidar and ('det' in self.tasks or 'occ' in self.tasks)
        lidar_bev = self._prepare_lidar(self._points_for_test(data.get('points'))) if need_lidar else None
        if 'det' in self.tasks:
            if lidar_bev is not None:
                data['lidar_bev'] = lidar_bev
            for result_dict, pts_bbox in zip(bbox_list, self.simple_test_pts(img_metas, **data)):
                result_dict['pts_bbox'] = pts_bbox
        if 'occ' in self.tasks:
            feats = data['img_feats']
            if feats.dim() == 6:
                feats = feats[:, -1]
            occ = self.forward_occ(feats, data, lidar_bev)
            for result_dict, occ_pred in zip(bbox_list, self.occ_head.get_occ(occ, img_metas)):
                result_dict['occ'] = occ_pred
        return bbox_list
