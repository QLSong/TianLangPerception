"""Occ3D 占用指标。

mIoU 在相机可见体素上计算，空类不进平均。
RayIoU 沿激光雷达射线取第一次命中的非空体素，深度差小于 1/2/4 米且类别相同算对。
网格和 vis_mini.crop_occ 一致。
"""
import math

import numpy as np
from numba import njit

OCC_NAMES = (
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'empty',
)
FREE = len(OCC_NAMES) - 1
PC_MIN = np.array([-40.0, -40.0, -1.0], dtype=np.float64)
VOXEL = 0.4
RAY_THS = (1.0, 2.0, 4.0)


def crop_occ(pred, grid, label_range):
    """和 vis_mini.crop_occ 相同的窗口。"""
    x0, _, dx = grid['x']
    y0, _, dy = grid['y']
    ix0 = int(round((label_range[0] - x0) / dx))
    ix1 = int(round((label_range[3] - x0) / dx))
    iy0 = int(round((label_range[1] - y0) / dy))
    iy1 = int(round((label_range[4] - y0) / dy))
    ix0, iy0 = max(ix0, 0), max(iy0, 0)
    ix1, iy1 = min(ix1, pred.shape[0]), min(iy1, pred.shape[1])
    return pred[ix0:ix1, iy0:iy1]


def generate_lidar_rays():
    """SparseOcc 的射线方向：方位角每 1°，俯仰覆盖 nuScenes 激光雷达视场。"""
    pitch_angles = []
    for k in range(10):
        pitch_angles.append(-(math.pi / 2 - math.atan(k + 1)))
    while pitch_angles[-1] < 0.21:
        pitch_angles.append(pitch_angles[-1] + (pitch_angles[-1] - pitch_angles[-2]))
    rays = []
    for pitch in pitch_angles:
        for azimuth_deg in range(0, 360):
            azimuth = math.radians(azimuth_deg)
            rays.append((
                math.cos(pitch) * math.cos(azimuth),
                math.cos(pitch) * math.sin(azimuth),
                math.sin(pitch),
            ))
    return np.asarray(rays, dtype=np.float64)


@njit
def _ray_cast(volume, origin, rays, pc_min, voxel, free, step, nstep):
    n = rays.shape[0]
    nx, ny, nz = volume.shape
    depth = np.empty(n, dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        depth[i] = np.inf
        labels[i] = free
        for s in range(nstep):
            dist = (s + 1) * step
            x = origin[0] + rays[i, 0] * dist
            y = origin[1] + rays[i, 1] * dist
            z = origin[2] + rays[i, 2] * dist
            ix = int(math.floor((x - pc_min[0]) / voxel))
            iy = int(math.floor((y - pc_min[1]) / voxel))
            iz = int(math.floor((z - pc_min[2]) / voxel))
            if ix < 0 or iy < 0 or iz < 0 or ix >= nx or iy >= ny or iz >= nz:
                continue
            lab = int(volume[ix, iy, iz])
            if lab != free:
                depth[i] = dist
                labels[i] = lab
                break
    return depth, labels


class OccMeter:
    """逐帧累加混淆矩阵和 RayIoU 计数，多卡上可以把数组相加。"""

    def __init__(self):
        n = len(OCC_NAMES)
        self.hist = np.zeros((n, n), dtype=np.int64)
        self.gt_cnt = np.zeros(n, dtype=np.int64)
        self.pred_cnt = np.zeros(n, dtype=np.int64)
        self.tp_cnt = np.zeros((len(RAY_THS), n), dtype=np.int64)
        self.rays = generate_lidar_rays()

    def update(self, pred, gt, mask, origin):
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        if gt.shape != pred.shape:
            if gt.size == np.prod(pred.shape):
                gt = gt.reshape(pred.shape)
            else:
                raise RuntimeError('occ pred {} 和标签 {} 对不上'.format(pred.shape, gt.shape))
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise RuntimeError('mask_camera {} 和标签 {} 对不上'.format(mask.shape, gt.shape))
        self._update_hist(pred, gt, mask)
        self._update_ray(pred, gt, origin)

    def _update_hist(self, pred, gt, mask):
        n = len(OCC_NAMES)
        gt_v = gt[mask].astype(np.int64).reshape(-1)
        pred_v = pred[mask].astype(np.int64).reshape(-1)
        keep = (gt_v >= 0) & (gt_v < n) & (pred_v >= 0) & (pred_v < n)
        flat = gt_v[keep] * n + pred_v[keep]
        self.hist += np.bincount(flat, minlength=n * n).reshape(n, n)

    def _update_ray(self, pred, gt, origin):
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        pred_depth, pred_label = _ray_cast(
            pred.astype(np.uint8), origin, self.rays, PC_MIN, VOXEL, FREE, VOXEL * 0.5, 400)
        gt_depth, gt_label = _ray_cast(
            gt.astype(np.uint8), origin, self.rays, PC_MIN, VOXEL, FREE, VOXEL * 0.5, 400)
        valid = gt_label != FREE
        pred_label = pred_label[valid]
        gt_label = gt_label[valid]
        error = np.abs(pred_depth[valid] - gt_depth[valid])
        for cls in range(FREE):
            gt_hit = gt_label == cls
            pred_hit = pred_label == cls
            self.gt_cnt[cls] += int(gt_hit.sum())
            self.pred_cnt[cls] += int(pred_hit.sum())
            both = gt_hit & pred_hit
            for index, threshold in enumerate(RAY_THS):
                self.tp_cnt[index, cls] += int((both & (error < threshold)).sum())

    def reduce_sum(self, other_hist, other_gt, other_pred, other_tp):
        self.hist += other_hist
        self.gt_cnt += other_gt
        self.pred_cnt += other_pred
        self.tp_cnt += other_tp

    def summary(self):
        tp = np.diag(self.hist).astype(np.float64)
        union = self.hist.sum(1) + self.hist.sum(0) - tp
        iou = np.full(len(OCC_NAMES), np.nan)
        valid_union = union > 0
        iou[valid_union] = tp[valid_union] / union[valid_union]
        miou = float(np.nanmean(iou[:FREE]))
        ray = []
        for index in range(len(RAY_THS)):
            denom = self.gt_cnt[:FREE] + self.pred_cnt[:FREE] - self.tp_cnt[index, :FREE]
            values = np.full(FREE, np.nan)
            ok = denom > 0
            values[ok] = self.tp_cnt[index, :FREE][ok] / denom[ok]
            ray.append(values)
        ray_mean = [float(np.nanmean(values)) for values in ray]
        lines = ['occ  mIoU 在相机可见体素上，空类不进平均']
        lines.append('{:<22} {:>8} {:>10} {:>10} {:>10}'.format(
            'class', 'IoU', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4'))
        for cls, name in enumerate(OCC_NAMES[:FREE]):
            lines.append('{:<22} {:8.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
                name, iou[cls], ray[0][cls], ray[1][cls], ray[2][cls]))
        lines.append('{:<22} {:8.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
            'MEAN', miou, ray_mean[0], ray_mean[1], ray_mean[2]))
        lines.append('mIoU {:.3f}  RayIoU {:.3f}'.format(miou, float(np.mean(ray_mean))))
        text = '\n'.join(lines)
        return dict(mIoU=miou, RayIoU=float(np.mean(ray_mean)),
                    RayIoU_1=ray_mean[0], RayIoU_2=ray_mean[1], RayIoU_4=ray_mean[2]), text
