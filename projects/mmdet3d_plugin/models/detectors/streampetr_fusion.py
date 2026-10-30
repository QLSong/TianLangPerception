"""相机 + 激光雷达多任务检测器。

检测走 StreamPETR 时序 query，每一层图像交叉注意力后再看一次 PointPillars BEV。
占用可以是平面投影加 UNet，OPUS 稀疏 query，或 GaussianOcc 体素。OPUS 打开雷达时，PointPillars BEV 先投到 query 通道，再按占据点采样后和图像特征拼在一起。
`use_camera` 关掉时不抽图像特征。`use_lidar` 关掉时不读点云。两路至少开一路。
`tasks` 取 det、occ 或 both；没打开的分支在 train() 里关掉梯度。
图像骨干可以是可训练的 ResNet-50，或冻结的 DINOv3 ConvNeXt。
交叉注意力用 PETRMultiheadAttention，占用 pooling 的导出符号是 bev_pool_v2。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models import build_backbone, build_head, build_neck
from mmseg.models.backbones.unet import UNet  # noqa: F401
from mmseg.models import build_backbone as build_mmseg_backbone

from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
from projects.mmdet3d_plugin.models.gaussian_occ.head import GaussianOccHead
from projects.mmdet3d_plugin.models.opus.opus_head import OPUSV1Head


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


def _is_occ_loss(key):
    return key.startswith('occ_') or 'loss_occ' in key or 'loss_voxel_' in key


def _keep_task_losses(losses, tasks):
    """单任务训练时丢掉另一个任务的 loss，避免被日志和总 loss 算进去。"""
    if 'det' in tasks and 'occ' in tasks:
        return losses
    det_marks = ('loss_cls', 'loss_bbox', 'loss_iou')
    kept = {}
    for key, value in losses.items():
        if _is_occ_loss(key):
            if 'occ' in tasks:
                kept[key] = value
            continue
        if any(mark in key for mark in det_marks):
            if 'det' in tasks:
                kept[key] = value
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


def _freeze_det_constants(head):
    """检测头里的常量不能跟着任务开关被重新打开。

    code_weights 乘在框回归的 L1 上。它一旦可训练，梯度就是绝对误差，
    优化器会把它一路减小；变成负数后 loss 变成负的，而且框的梯度反向，
    误差越大 loss 越负。
    """
    if head is None:
        return
    for name in ('code_weights', 'match_costs', 'pc_range', 'position_range', 'coords_d'):
        param = getattr(head, name, None)
        if torch.is_tensor(param):
            param.requires_grad = False
    points = getattr(head, 'pseudo_reference_points', None)
    if points is not None:
        points.weight.requires_grad = False


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
                 use_camera=True,
                 freeze_img_backbone=None,
                 freeze_lidar_backbone=False,
                 det_from_view=False,
                 lidar_bev_channels=384,
                 **kwargs):
        super().__init__(**kwargs)
        self.tasks = parse_tasks(tasks)
        self.use_lidar = bool(use_lidar)
        self.use_camera = bool(use_camera)
        # None 表示沿用图像骨干自己的 frozen。雷达默认不冻。
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_lidar_backbone = bool(freeze_lidar_backbone)
        if not self.use_lidar and not self.use_camera:
            raise ValueError('use_lidar 和 use_camera 至少打开一个')
        # 检测交叉注意力读 PlaneViewTransformer 的 BEV，不再读原图 token。
        self.det_from_view = bool(det_from_view)
        self.lidar_backbone = build_backbone(pts_backbone) if pts_backbone else None
        self.img_view_transformer = build_neck(img_view_transformer) if img_view_transformer else None
        self.bev_backbone = build_mmseg_backbone(bev_backbone) if bev_backbone else None
        bev_out = 64
        if bev_backbone is not None:
            bev_out = bev_backbone.get('base_channels', 64)
        occ_in = occ_head.get('in_dim', 256) if occ_head else 256
        self.bev_neck = nn.Conv2d(bev_out, occ_in, kernel_size=1) if self.bev_backbone else None
        self.occ_head = self._build_occ_head(occ_head) if occ_head else None
        self.lidar_opus_proj = None
        if (self.lidar_backbone is not None and isinstance(self.occ_head, OPUSV1Head)
                and getattr(self.occ_head.transformer, 'with_pts', False)):
            self.lidar_opus_proj = ConvModule(
                lidar_bev_channels, self.occ_head.embed_dims, kernel_size=3,
                stride=1, padding=1, bias=True, conv_cfg=dict(type='Conv2d'))
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

    def _img_backbone_frozen(self):
        if self.freeze_img_backbone is not None:
            return bool(self.freeze_img_backbone)
        return bool(getattr(self.img_backbone, 'frozen', False))

    def _apply_task_freeze(self):
        train_det = 'det' in self.tasks
        train_occ = 'occ' in self.tasks
        img_frozen = self._img_backbone_frozen()
        if (self.img_backbone is not None and hasattr(self.img_backbone, 'frozen')
                and self.freeze_img_backbone is not None):
            self.img_backbone.frozen = img_frozen
        _set_requires_grad(self.img_backbone, self.use_camera and not img_frozen)
        if img_frozen and self.img_backbone is not None:
            self.img_backbone.eval()
        _set_requires_grad(self.img_neck, self.use_camera)
        lidar_train = self.use_lidar and (train_det or train_occ) and not self.freeze_lidar_backbone
        _set_requires_grad(self.lidar_backbone, lidar_train)
        if self.freeze_lidar_backbone and self.lidar_backbone is not None:
            self.lidar_backbone.eval()
        _set_requires_grad(self.lidar_occ_fuse, self.use_lidar and train_occ)
        _set_requires_grad(self.lidar_opus_proj, self.use_lidar and train_occ)
        _set_requires_grad(self.pts_bbox_head, train_det)
        _freeze_det_constants(self.pts_bbox_head)
        _set_requires_grad(
            self.img_view_transformer,
            self.use_camera and (train_occ or (train_det and self.det_from_view)))
        _set_requires_grad(self.bev_backbone, train_occ)
        _set_requires_grad(self.bev_neck, train_occ)
        _set_requires_grad(self.occ_head, train_occ)

    def train(self, mode=True):
        if (self.img_backbone is not None and hasattr(self.img_backbone, 'frozen')
                and self.freeze_img_backbone is not None):
            self.img_backbone.frozen = bool(self.freeze_img_backbone)
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

    def _occ_hw(self):
        grid = self.img_view_transformer.grid_config
        nx = int(round((grid['x'][1] - grid['x'][0]) / grid['x'][2]))
        ny = int(round((grid['y'][1] - grid['y'][0]) / grid['y'][2]))
        return ny, nx

    def _needs_lidar(self):
        """GaussianOcc 不读点云。占用只开这一支时不再跑 PointPillars。"""
        if not self.use_lidar:
            return False
        if 'det' in self.tasks:
            return True
        return 'occ' in self.tasks and not isinstance(self.occ_head, GaussianOccHead)

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
        if self.freeze_lidar_backbone:
            with torch.no_grad():
                return self.lidar_backbone(pillars, coords)
        return self.lidar_backbone(pillars, coords)

    def _lss_inputs(self, img_feats, extrinsics, intrinsics, distortion=None):
        """外参是自车到相机。LSS 的 BEV 落在自车坐标，x 朝前、y 朝左。

        distortion 是 (B, N, 8) 时，IPM 在乘内参前做 rational 畸变。
        """
        b, n = img_feats.shape[:2]
        device = img_feats.device
        if extrinsics.dim() == 5:
            extrinsics = extrinsics[:, -1]
            intrinsics = intrinsics[:, -1]
            if distortion is not None and distortion.dim() == 4:
                distortion = distortion[:, -1]
        sensor2ego = torch.linalg.inv(extrinsics)
        ego2global = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(b, n, 1, 1)
        cam2imgs = intrinsics[..., :3, :3]
        post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
        post_trans = torch.zeros(b, n, 3, device=device)
        bda = torch.eye(3, device=device).unsqueeze(0).repeat(b, 1, 1)
        packed = [img_feats, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda]
        if distortion is not None:
            packed.append(distortion.to(device=device, dtype=img_feats.dtype))
        return packed

    def _build_occ_head(self, occ_head):
        occ_head = occ_head.copy()
        if occ_head.get('type') == 'OPUSV1Head':
            if self.train_cfg is not None and self.train_cfg.get('occ') is not None:
                occ_head['train_cfg'] = self.train_cfg.get('occ')
            if self.test_cfg is not None and self.test_cfg.get('occ') is not None:
                occ_head['test_cfg'] = self.test_cfg.get('occ')
        return build_head(occ_head)

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        if not isinstance(getattr(self, 'occ_head', None), OPUSV1Head):
            return super().extract_img_feat(img, len_queue, training_mode)
        batch = img.size(0)
        if img.dim() == 6:
            img = img.flatten(1, 2)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            _, n, c, h, w = img.size()
            img = img.reshape(batch * n, c, h, w)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        mlvl = []
        for feat in img_feats:
            bn, channels, height, width = feat.shape
            mlvl.append(feat.view(batch, bn // batch, channels, height, width))
        self._opus_mlvl = mlvl
        level = img_feats[self.position_level]
        bn, channels, height, width = level.shape
        views = bn // batch // len_queue
        if self.training or training_mode:
            return level.view(batch, len_queue, views, channels, height, width)
        return level.view(batch, views, channels, height, width)

    def _opus_metas(self, img_metas, lidar2img):
        if img_metas and isinstance(img_metas[0], (list, tuple)):
            frame_metas = list(img_metas[-1])
        else:
            frame_metas = list(img_metas)
        mats = lidar2img
        if torch.is_tensor(mats):
            if mats.dim() == 5:
                mats = mats[:, -1]
            mats = mats.detach().float().cpu().numpy()
        metas = []
        for i, meta in enumerate(frame_metas):
            meta = dict(meta)
            meta['ego2img'] = np.asarray(mats[i], dtype=np.float32)
            meta['ego2occ'] = np.eye(4, dtype=np.float32)
            meta['ego2lidar'] = np.eye(4, dtype=np.float32)
            metas.append(meta)
        return metas

    def _opus_dense(self, result):
        shape = tuple(int(v) for v in self.occ_head.voxel_num.tolist())
        occ = np.full(shape, int(self.occ_head.empty_label), dtype=np.uint8)
        loc = result['occ_loc']
        if len(loc):
            occ[loc[:, 0], loc[:, 1], loc[:, 2]] = result['sem_pred']
        return occ

    def _opus_pts(self, lidar_bev):
        if self.lidar_opus_proj is None:
            return None
        if lidar_bev is None:
            raise RuntimeError('OPUS lidar sampling is on but the batch has no lidar BEV')
        return self.lidar_opus_proj(lidar_bev)

    def _camera_bev(self, img_feats, extrinsics, intrinsics, distortion=None):
        """PlaneViewTransformer 的输出。有时间维时逐帧投，返回 (B, T, C, Y, X)。"""
        if self.img_view_transformer is None:
            raise RuntimeError('需要 img_view_transformer 才能得到相机 BEV')
        if img_feats.dim() == 5:
            bev, _ = self.img_view_transformer(
                self._lss_inputs(img_feats, extrinsics, intrinsics, distortion))
            return bev
        if img_feats.dim() != 6:
            raise RuntimeError('图像特征维度是 {}'.format(tuple(img_feats.shape)))
        frames = []
        for t in range(img_feats.size(1)):
            ext = extrinsics[:, t] if extrinsics.dim() == 5 else extrinsics
            intr = intrinsics[:, t] if intrinsics.dim() == 5 else intrinsics
            dist = None
            if distortion is not None:
                dist = distortion[:, t] if distortion.dim() == 4 else distortion
            bev, _ = self.img_view_transformer(
                self._lss_inputs(img_feats[:, t], ext, intr, dist))
            frames.append(bev)
        return torch.stack(frames, 1)

    def _attach_det_bev(self, img_feats, data):
        if not self.det_from_view:
            return None
        camera_bev = self._camera_bev(
            img_feats, data['extrinsics'], data['intrinsics'], data.get('distortion'))
        data['img_bev'] = camera_bev
        return camera_bev

    def forward_occ(self, img_feats, data, lidar_bev=None, camera_bev=None):
        if self.img_view_transformer is None:
            return None
        if not self.use_camera:
            if isinstance(self.occ_head, (GaussianOccHead, OPUSV1Head)):
                raise RuntimeError('单雷达占用只支持 BEVOCCHead2D')
            if lidar_bev is None or self.lidar_occ_fuse is None:
                raise RuntimeError('单雷达占用没有雷达 BEV')
            canvas = lidar_bev.new_zeros(lidar_bev.shape[0], 1, *self._occ_hw())
            aligned = align_bev(
                lidar_bev, self._lidar_xy_range(), canvas, self._camera_xy_range())
            bev = self.lidar_occ_fuse(aligned)
        else:
            if camera_bev is None:
                camera_bev = self._camera_bev(
                    img_feats, data['extrinsics'], data['intrinsics'], data.get('distortion'))
            if camera_bev.dim() == 5:
                camera_bev = camera_bev[:, -1]
            bev = camera_bev
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
        if self.pts_bbox_head is not None:
            self.pts_bbox_head.attend_image = self.use_camera
        if self.use_camera:
            t = data['img'].size(1)
            rec_img = data['img'][:, -self.num_frame_backbone_grads:]
            data['img_feats'] = self.extract_feat(rec_img, self.num_frame_backbone_grads)
        elif data.get('img') is not None:
            img = data['img']
            t = img.size(1)
            data['img_feats'] = img.new_zeros(img.shape[0], t, 1, 1, 1)
        else:
            t = 1
            data['img_feats'] = None
        losses = dict()
        lidar_bev = self._prepare_lidar(points) if self._needs_lidar() else None
        camera_bev = (
            self._attach_det_bev(data['img_feats'], data)
            if train_det and self.use_camera else None)
        if train_det and data.get('img') is not None:
            if lidar_bev is not None:
                data['lidar_bev'] = lidar_bev.unsqueeze(1).repeat(1, t, 1, 1, 1)
            losses.update(self.obtain_history_memory(
                gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, img_metas,
                centers2d, depths, gt_bboxes_ignore, **data))
        elif train_det:
            if lidar_bev is not None:
                data['lidar_bev'] = lidar_bev
            loss = self.forward_pts_train(
                gt_bboxes_3d[0], gt_labels_3d[0], gt_bboxes[0], gt_labels[0],
                img_metas[0], centers2d[0], depths[0],
                requires_grad=True, return_losses=True, **data)
            if loss is not None:
                losses.update(loss)
        if train_occ:
            if not self.use_camera and isinstance(self.occ_head, (GaussianOccHead, OPUSV1Head)):
                raise RuntimeError('单雷达占用只支持 BEVOCCHead2D')
            if isinstance(self.occ_head, GaussianOccHead) and self.occ_head.supervise != 'gt':
                missing = [name for name in (
                    'gs_color', 'gs_color_m1', 'gs_color_p1', 'gs_semantic',
                    'gs_cam_T_m1', 'gs_cam_T_p1') if data.get(name) is None]
                if missing:
                    raise RuntimeError(
                        'GaussianOcc 自监督缺少 {}。训练集需要 gsocc_supervise=True'.format(
                            ', '.join(missing)))
                losses.update(self.occ_head.loss_self(
                    data['img_feats'], data['extrinsics'], data['intrinsics'],
                    data['gs_color'], data['gs_color_m1'], data['gs_color_p1'],
                    data['gs_cam_T_m1'], data['gs_cam_T_p1'], data['gs_semantic']))
            elif voxel_semantics is None:
                raise RuntimeError('task includes occ but the batch has no voxel_semantics')
            elif isinstance(self.occ_head, OPUSV1Head):
                if mask_camera is None:
                    mask_camera = torch.ones_like(voxel_semantics)
                preds = self.occ_head(
                    self._opus_mlvl, self._opus_metas(img_metas, data['lidar2img']),
                    self._opus_pts(lidar_bev))
                for key, value in self.occ_head.loss(voxel_semantics, mask_camera, preds).items():
                    losses['occ_' + key] = value
            elif isinstance(self.occ_head, GaussianOccHead):
                occ = self.occ_head(
                    data['img_feats'], data['extrinsics'], data['intrinsics'],
                    (data['img'].shape[-2], data['img'].shape[-1]))
                losses.update(self.occ_head.loss(occ, voxel_semantics, mask_camera))
            else:
                last_bev = None
                if camera_bev is not None:
                    last_bev = camera_bev[:, -1] if camera_bev.dim() == 5 else camera_bev
                feats = data['img_feats']
                if feats is not None:
                    feats = feats[:, -1]
                occ = self.forward_occ(feats, data, lidar_bev, camera_bev=last_bev)
                losses.update(self.occ_head.loss(occ, voxel_semantics, mask_camera))
        return _keep_task_losses(losses, self.tasks)

    def _points_for_test(self, points):
        """测试前向把点云收成 (B, N, C)，还原成 encode_lidar 要的逐帧列表。"""
        points = self._as_points(points)
        if torch.is_tensor(points) and points.dim() == 3:
            points = [points[i] for i in range(points.size(0))]
        return points

    def forward_pts_train(self, gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels,
                          img_metas, centers2d, depths, requires_grad=True,
                          return_losses=False, **data):
        if self.use_camera:
            return super().forward_pts_train(
                gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, img_metas,
                centers2d, depths, requires_grad=requires_grad,
                return_losses=return_losses, **data)
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(None, img_metas, None, **data)
            self.train()
        else:
            outs = self.pts_bbox_head(None, img_metas, None, **data)
        if not return_losses:
            return None
        return self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)

    def simple_test_pts(self, img_metas, **data):
        if self.use_camera:
            return super().simple_test_pts(img_metas, **data)
        ref = data.get('lidar_bev')
        if ref is None:
            ref = data['points'][0]
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = ref.new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = ref.new_ones(1)
        outs = self.pts_bbox_head(None, img_metas, None, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        return [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list]

    def simple_test(self, img_metas, **data):
        if self.pts_bbox_head is not None:
            self.pts_bbox_head.attend_image = self.use_camera
        if self.use_camera:
            data['img_feats'] = self.extract_img_feat(data['img'], 1)
        else:
            data['img_feats'] = None
        bbox_list = [dict() for _ in range(len(img_metas))]
        need_lidar = self._needs_lidar()
        lidar_bev = self._prepare_lidar(self._points_for_test(data.get('points'))) if need_lidar else None
        camera_bev = (
            self._attach_det_bev(data['img_feats'], data)
            if 'det' in self.tasks and self.use_camera else None)
        if 'det' in self.tasks:
            if lidar_bev is not None:
                data['lidar_bev'] = lidar_bev
            for result_dict, pts_bbox in zip(bbox_list, self.simple_test_pts(img_metas, **data)):
                result_dict['pts_bbox'] = pts_bbox
        if 'occ' in self.tasks:
            if not self.use_camera and isinstance(self.occ_head, (GaussianOccHead, OPUSV1Head)):
                raise RuntimeError('单雷达占用只支持 BEVOCCHead2D')
            if isinstance(self.occ_head, OPUSV1Head):
                metas = self._opus_metas(img_metas, data['lidar2img'])
                preds = self.occ_head(self._opus_mlvl, metas, self._opus_pts(lidar_bev))
                occ_preds = [self._opus_dense(item) for item in self.occ_head.get_occ(preds, metas)]
            elif isinstance(self.occ_head, GaussianOccHead):
                occ = self.occ_head(
                    data['img_feats'], data['extrinsics'], data['intrinsics'],
                    (data['img'].shape[-2], data['img'].shape[-1]))
                occ_preds = self.occ_head.get_occ(occ, img_metas)
            else:
                feats = data['img_feats']
                if feats is not None and feats.dim() == 6:
                    feats = feats[:, -1]
                occ = self.forward_occ(feats, data, lidar_bev, camera_bev=camera_bev)
                occ_preds = self.occ_head.get_occ(occ, img_metas)
            for result_dict, occ_pred in zip(bbox_list, occ_preds):
                result_dict['occ'] = occ_pred
        return bbox_list
