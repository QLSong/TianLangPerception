"""在小训练集上按时间顺序自测，并把预测和真值画到同一张图。

六路相机按环视排列，3D 框投到每一路图像上：绿是真值，按类别上色的是预测。
下面先画俯视图，再画从后上方看的 3D 透视。必须按样本顺序跑，时序记忆才和训练时一致。

示例：
python tools/vis_mini.py \\
  projects/configs/TianLangEyes/streampetr_r50_pp_lss_nus_mini.py \\
  work_dirs/streampetr_r50_pp_lss_nus_mini/latest.pth \\
  --out-dir work_dirs/vis_mini
"""
import argparse
import importlib
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
    convert_egopose_to_matrix_numpy, invert_matrix_egopose_numpy,
    lidar_to_ego_matrix, transform_points_lidar_to_ego)
from projects.mmdet3d_plugin.datasets.pipelines.loading import _index_occ_root

OCC_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'empty',
]
OCC_COLORS = np.array([
    [0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 127, 80], [255, 158, 0],
    [233, 150, 70], [255, 61, 99], [0, 0, 230], [47, 79, 79], [255, 140, 0],
    [255, 99, 71], [0, 207, 191], [175, 0, 75], [75, 0, 75], [112, 180, 60],
    [222, 184, 135], [0, 175, 0], [230, 230, 230],
], dtype=np.uint8)
DET_COLORS = [
    (255, 158, 0), (255, 99, 71), (233, 150, 70), (255, 127, 80), (255, 140, 0),
    (112, 128, 144), (255, 61, 99), (220, 20, 60), (0, 0, 230), (47, 79, 79),
]
EMPTY = 17
CAM_LAYOUT = (
    ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'),
    ('CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'),
)
IMG_W, IMG_H = 640, 360
PLAN_SIZE = 640
BOX_EDGES = (
    (0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7),
)
# 底面不画。顺序绕着朝外的面，明暗按从上往下看。
BOX_FACES = (
    ((0, 3, 2, 1), 0.62),
    ((4, 5, 6, 7), 0.45),
    ((0, 1, 5, 4), 0.75),
    ((3, 7, 6, 2), 0.75),
    ((1, 5, 6, 2), 1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description='小训练集自测可视化')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out-dir', default='work_dirs/vis_mini')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--max-frames', type=int, default=0, help='0 表示全部')
    return parser.parse_args()


def import_plugin(cfg):
    plugin_dir = cfg.plugin_dir
    module = plugin_dir.replace('/', '.').rstrip('.')
    if module.endswith('.'):
        module = module[:-1]
    importlib.import_module(module)


def world_to_pixel(x, y, bounds, width, height):
    xmin, ymin, xmax, ymax = bounds
    px = (ymax - y) / (ymax - ymin) * (width - 1)
    py = (xmax - x) / (xmax - xmin) * (height - 1)
    return np.stack([px, py], axis=-1)


def _text(canvas, text, org, color):
    org = (int(np.clip(org[0], 4, canvas.shape[1] - 36)), int(np.clip(org[1], 16, canvas.shape[0] - 6)))
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_axes(canvas, bounds, axis_len=10.0):
    """自车坐标：+x 朝前（图上方，红），+y 朝左（蓝）。"""
    h, w = canvas.shape[:2]
    xmin, ymin, xmax, ymax = bounds
    if not (xmin < 0 < xmax and ymin < 0 < ymax):
        return
    span = min(xmax - xmin, ymax - ymin)
    axis_len = min(axis_len, span * 0.35)

    def pix(x, y):
        return world_to_pixel(np.array([x]), np.array([y]), bounds, w, h)[0]

    origin = pix(0.0, 0.0)
    x_ends = (pix(xmin, 0.0), pix(xmax, 0.0))
    y_ends = (pix(0.0, ymin), pix(0.0, ymax))
    for p0, p1 in (x_ends, y_ends):
        a = tuple(np.round(p0).astype(int))
        b = tuple(np.round(p1).astype(int))
        cv2.line(canvas, a, b, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(canvas, a, b, (40, 40, 40), 1, cv2.LINE_AA)
    x_tip = pix(axis_len, 0.0)
    y_tip = pix(0.0, axis_len)
    o = tuple(np.round(origin).astype(int))
    cv2.arrowedLine(canvas, o, tuple(np.round(x_tip).astype(int)), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, o, tuple(np.round(x_tip).astype(int)), (0, 0, 220), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, o, tuple(np.round(y_tip).astype(int)), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, o, tuple(np.round(y_tip).astype(int)), (220, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(canvas, o, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, o, 3, (40, 40, 40), -1, cv2.LINE_AA)
    _text(canvas, '+x', x_tip + np.array([6, -6]), (0, 0, 220))
    _text(canvas, '+y', y_tip + np.array([6, 16]), (220, 0, 0))


def draw_points(canvas, points, bounds):
    h, w = canvas.shape[:2]
    keep = (
        (points[:, 0] >= bounds[0]) & (points[:, 0] < bounds[2])
        & (points[:, 1] >= bounds[1]) & (points[:, 1] < bounds[3]))
    xy = points[keep][::4, :2]
    if xy.shape[0] == 0:
        return
    pix = world_to_pixel(xy[:, 0], xy[:, 1], bounds, w, h).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, h - 1)
    canvas[pix[:, 1], pix[:, 0]] = (180, 180, 180)


def draw_boxes(canvas, boxes, labels, bounds, color=None, scores=None):
    if boxes is None or len(boxes) == 0:
        return
    h, w = canvas.shape[:2]
    corners = boxes.corners.numpy()[:, [0, 3, 7, 4], :2]
    for i, corner in enumerate(corners):
        if labels is not None and int(labels[i]) < 0:
            continue
        pix = world_to_pixel(corner[:, 0], corner[:, 1], bounds, w, h).astype(np.int32)
        shade = color
        if shade is None:
            rgb = DET_COLORS[int(labels[i]) % len(DET_COLORS)]
            shade = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.polylines(canvas, [pix.reshape(-1, 1, 2)], True, shade, 2, cv2.LINE_AA)
        if scores is not None:
            cv2.putText(
                canvas, '{:.2f}'.format(float(scores[i])),
                tuple(pix[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, shade, 1, cv2.LINE_AA)


def as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def camera_ego2img(cam_info, lidar2ego):
    """自车坐标到图像。点云文件仍是激光雷达坐标，先乘 lidar2ego 再投影。"""
    cam2lidar = convert_egopose_to_matrix_numpy(
        cam_info['sensor2lidar_rotation'], cam_info['sensor2lidar_translation'])
    lidar2cam = invert_matrix_egopose_numpy(cam2lidar)
    ego2cam = lidar2cam @ invert_matrix_egopose_numpy(lidar2ego)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:3, :3] = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
    return viewpad @ ego2cam


def scale_lidar2img(lidar2img, src_hw, dst_hw):
    scaled = np.array(lidar2img, dtype=np.float64, copy=True)
    scaled[0] *= dst_hw[1] / src_hw[1]
    scaled[1] *= dst_hw[0] / src_hw[0]
    return scaled


def _clip_edge(p0, p1, near=0.1):
    z0, z1 = p0[2], p1[2]
    if z0 <= near and z1 <= near:
        return None
    if z0 <= near or z1 <= near:
        if z0 > z1:
            p0, p1 = p1, p0
        t = (near - p0[2]) / (p1[2] - p0[2])
        p0 = p0 + t * (p1 - p0)
    uv0 = p0[:2] / p0[2]
    uv1 = p1[:2] / p1[2]
    if not (np.isfinite(uv0).all() and np.isfinite(uv1).all()):
        return None
    uv0 = np.clip(uv0, -1e5, 1e5)
    uv1 = np.clip(uv1, -1e5, 1e5)
    return (
        (int(round(uv0[0])), int(round(uv0[1]))),
        (int(round(uv1[0])), int(round(uv1[1]))),
        float(min(p0[2], p1[2])),
    )


def draw_boxes_on_image(image, boxes, labels, lidar2img, class_names, color=None, scores=None):
    if boxes is None or len(boxes) == 0:
        return
    corners = as_numpy(boxes.corners)
    labels = as_numpy(labels)
    h, w = image.shape[:2]
    hom = np.concatenate([corners, np.ones(corners.shape[:2] + (1,))], axis=-1)
    cam = hom @ np.asarray(lidar2img, dtype=np.float64).T
    for i in range(cam.shape[0]):
        if labels is not None and int(labels[i]) < 0:
            continue
        if color is None:
            rgb = DET_COLORS[int(labels[i]) % len(DET_COLORS)]
            shade = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        else:
            shade = color
        anchor = None
        anchor_depth = None
        for a, b in BOX_EDGES:
            edge = _clip_edge(cam[i, a], cam[i, b])
            if edge is None:
                continue
            p0, p1, depth = edge
            inside, q0, q1 = cv2.clipLine((0, 0, w, h), p0, p1)
            if not inside:
                continue
            cv2.line(image, q0, q1, shade, 2, cv2.LINE_AA)
            if anchor_depth is None or depth < anchor_depth:
                anchor_depth = depth
                anchor = q0
        if anchor is None or scores is None:
            continue
        name = class_names[int(labels[i])] if class_names else str(int(labels[i]))
        cv2.putText(
            image, '{} {:.2f}'.format(name, float(scores[i])),
            anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, shade, 1, cv2.LINE_AA)


def stitch_rows(rows):
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] == width:
            padded.append(row)
            continue
        padded.append(cv2.copyMakeBorder(
            row, 0, 0, 0, width - row.shape[1],
            cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    return np.concatenate(padded, axis=0)


def collapse_occ(occ):
    vis = np.full(occ.shape[:2], EMPTY, dtype=np.int32)
    for z in range(occ.shape[2]):
        layer = occ[:, :, z]
        hit = layer != EMPTY
        vis[hit] = layer[hit]
    return vis


def colorize_occ(occ, size, bounds):
    vis = collapse_occ(occ)
    image = OCC_COLORS[vis][::-1, ::-1]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    draw_axes(image, bounds)
    return image


def crop_occ(pred, grid, label_range):
    x0, _, dx = grid['x']
    y0, _, dy = grid['y']
    ix0 = int(round((label_range[0] - x0) / dx))
    ix1 = int(round((label_range[3] - x0) / dx))
    iy0 = int(round((label_range[1] - y0) / dy))
    iy1 = int(round((label_range[4] - y0) / dy))
    ix0, iy0 = max(ix0, 0), max(iy0, 0)
    ix1, iy1 = min(ix1, pred.shape[0]), min(iy1, pred.shape[1])
    bounds = (x0 + ix0 * dx, y0 + iy0 * dy, x0 + ix1 * dx, y0 + iy1 * dy)
    return pred[ix0:ix1, iy0:iy1], bounds


def _look_at(eye, target):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, forward], axis=0)


def _project(points, eye, rot):
    cam = (points - eye) @ rot.T
    depth = cam[:, 2]
    valid = depth > 0.5
    depth_safe = np.clip(depth, 0.5, None)
    u = cam[:, 0] / depth_safe
    v = -cam[:, 1] / depth_safe
    return np.stack([u, v], axis=1), depth, valid


def _fit_camera(xy_bounds, z_bounds, size):
    """相机在范围的后上方，看向地面中心。画面按体范围撑满。"""
    x0, y0, x1, y1 = [float(v) for v in xy_bounds]
    z0, z1 = [float(v) for v in z_bounds]
    eye = np.array([x0 - 28.0, y0 + 18.0, z1 + 42.0], dtype=np.float64)
    target = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, z0], dtype=np.float64)
    rot = _look_at(eye, target)
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=np.float64)
    projected, _, valid = _project(corners, eye, rot)
    projected = projected[valid]
    if projected.size == 0:
        span = np.array([1.0, 1.0])
        center = np.zeros(2)
    else:
        span = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1e-3)
        center = projected.mean(axis=0)
    margin = 36
    scale = min((size - 2 * margin) / span[0], (size - 2 * margin) / span[1])

    def to_pixel(uv):
        pix = (uv - center) * scale
        pix[:, 0] += size * 0.5
        pix[:, 1] += size * 0.55
        return pix

    return eye, rot, to_pixel


def _draw_persp_line(canvas, p0, p1, eye, rot, to_pixel, color, thickness):
    pts = np.stack([p0, p1], axis=0).astype(np.float64)
    uv, _, ok = _project(pts, eye, rot)
    if not ok.all():
        return
    pix = np.round(to_pixel(uv)).astype(np.int32)
    cv2.line(
        canvas,
        (int(pix[0, 0]), int(pix[0, 1])),
        (int(pix[1, 0]), int(pix[1, 1])),
        color, thickness, cv2.LINE_AA)


def _draw_persp_ground(canvas, xy_bounds, z0, eye, rot, to_pixel):
    x0, y0, x1, y1 = [float(v) for v in xy_bounds]
    for value in np.arange(np.ceil(x0 / 10.0) * 10.0, x1, 10.0):
        _draw_persp_line(canvas, [value, y0, z0], [value, y1, z0], eye, rot, to_pixel, (226, 226, 226), 1)
    for value in np.arange(np.ceil(y0 / 10.0) * 10.0, y1, 10.0):
        _draw_persp_line(canvas, [x0, value, z0], [x1, value, z0], eye, rot, to_pixel, (226, 226, 226), 1)


def _draw_persp_axes(canvas, eye, rot, to_pixel):
    origin = np.zeros(3, dtype=np.float64)
    for tip, text, color in (
            (np.array([10.0, 0.0, 0.0]), 'x', (0, 0, 220)),
            (np.array([0.0, 10.0, 0.0]), 'y', (220, 0, 0)),
            (np.array([0.0, 0.0, 4.0]), 'z', (0, 160, 0))):
        _draw_persp_line(canvas, origin, tip, eye, rot, to_pixel, color, 2)
        uv, _, ok = _project(tip.reshape(1, 3), eye, rot)
        if not ok[0]:
            continue
        pix = to_pixel(uv)[0]
        cv2.putText(
            canvas, text, (int(pix[0]) + 4, int(pix[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _shade(color, factor):
    if factor == 1.0:
        return color
    return tuple(int(channel * factor) for channel in color)


def _poly_inside(poly, shape):
    h, w = shape[:2]
    return not (
        poly[:, 0].max() < 0 or poly[:, 1].max() < 0
        or poly[:, 0].min() >= w or poly[:, 1].min() >= h)


def _blend_poly(canvas, poly, color, alpha):
    if not _poly_inside(poly, canvas.shape):
        return
    poly = np.round(poly).astype(np.int32)
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    x0 = max(int(x0), 0)
    y0 = max(int(y0), 0)
    x1 = min(int(x1) + 1, canvas.shape[1])
    y1 = min(int(y1) + 1, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    over = region.copy()
    cv2.fillConvexPoly(over, poly - np.array([x0, y0]), color, cv2.LINE_AA)
    mask = np.zeros(region.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, poly - np.array([x0, y0]), 255, cv2.LINE_AA)
    weight = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    canvas[y0:y1, x0:x1] = (over * weight + region * (1.0 - weight)).astype(np.uint8)


def render_det_3d(points, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                  xy_bounds, z_bounds, size=PLAN_SIZE):
    """检测框的透视。相机在后上方，+x 朝前，+y 朝左，+z 朝上。"""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    eye, rot, to_pixel = _fit_camera(xy_bounds, z_bounds, size)
    z0 = float(z_bounds[0])
    _draw_persp_ground(canvas, xy_bounds, z0, eye, rot, to_pixel)
    if points is not None and len(points) > 0:
        xy = points[::4][:, :3]
        uv, _, ok = _project(xy.astype(np.float64), eye, rot)
        if ok.any():
            pix = np.round(to_pixel(uv[ok])).astype(np.int32)
            h, w = canvas.shape[:2]
            inside = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
            pix = pix[inside]
            canvas[pix[:, 1], pix[:, 0]] = (186, 186, 186)

    layers = []
    for boxes, labels, scores, color in (
            (gt_boxes, gt_labels, None, (0, 180, 0)),
            (pred_boxes, pred_labels, pred_scores, None)):
        corners = None if boxes is None or len(boxes) == 0 else as_numpy(boxes.corners)
        if corners is None or len(corners) == 0:
            continue
        labels = as_numpy(labels)
        scores = as_numpy(scores)
        for i, corner in enumerate(corners):
            if labels is not None and int(labels[i]) < 0:
                continue
            if color is None:
                rgb = DET_COLORS[int(labels[i]) % len(DET_COLORS)]
                shade = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            else:
                shade = color
            layers.append((corner.astype(np.float64), shade, None if scores is None else float(scores[i])))

    faces = []
    for corner, shade, score in layers:
        uv, depth, valid = _project(corner, eye, rot)
        pix = to_pixel(uv)
        for index, factor in BOX_FACES:
            index = list(index)
            if not valid[index].all():
                continue
            faces.append((
                float(depth[index].mean()),
                pix[index],
                _shade(shade, factor),
                shade,
            ))
    faces.sort(key=lambda item: -item[0])
    for _, poly, face_color, _ in faces:
        _blend_poly(canvas, poly, face_color, 0.9)
    for corner, shade, score in layers:
        uv, depth, valid = _project(corner, eye, rot)
        pix = np.round(to_pixel(uv)).astype(np.int32)
        h, w = canvas.shape[:2]
        for a, b in BOX_EDGES:
            if not (valid[a] and valid[b]):
                continue
            ok, p0, p1 = cv2.clipLine(
                (0, 0, w, h), (int(pix[a, 0]), int(pix[a, 1])), (int(pix[b, 0]), int(pix[b, 1])))
            if ok:
                cv2.line(canvas, p0, p1, shade, 1, cv2.LINE_AA)
        if score is None:
            continue
        top_index = [i for i in (1, 2, 5, 6) if valid[i]]
        if not top_index:
            continue
        top = pix[top_index]
        anchor = top[np.argmin(top[:, 1])]
        cv2.putText(
            canvas, '{:.2f}'.format(score),
            (int(anchor[0]), int(anchor[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, shade, 1, cv2.LINE_AA)
    _draw_persp_axes(canvas, eye, rot, to_pixel)
    return canvas


def render_occ_3d(occ, xy_bounds, z_bounds, size=PLAN_SIZE):
    """从后上方看占用体素。+x 朝前，+y 朝左，+z 朝上。"""
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = [float(v) for v in xy_bounds]
    z0, z1 = [float(v) for v in z_bounds]
    nx, ny, nz = occ.shape
    dx = (x1 - x0) / nx
    dy = (y1 - y0) / ny
    dz = (z1 - z0) / max(nz, 1)
    eye, rot, to_pixel = _fit_camera(xy_bounds, z_bounds, size)
    _draw_persp_ground(canvas, xy_bounds, z0, eye, rot, to_pixel)
    known = (occ < len(OCC_COLORS)) & (occ != EMPTY)
    padded = np.pad(known, 1, constant_values=False)
    # 相机在 -x、-y 一侧，只画朝向相机的外露面。
    face_masks = (
        (known & ~padded[:-2, 1:-1, 1:-1], ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)), 0.62),
        (known & ~padded[1:-1, :-2, 1:-1], ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), 0.82),
        (known & ~padded[1:-1, 1:-1, 2:], ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), 1.0),
    )
    hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5
    half = (hx, hy, hz)
    face_list = []
    color_list = []
    for mask, corners, shade in face_masks:
        xs, ys, zs = np.nonzero(mask)
        if xs.size == 0:
            continue
        base = np.stack([
            x0 + (xs.astype(np.float64) + 0.5) * dx,
            y0 + (ys.astype(np.float64) + 0.5) * dy,
            z0 + (zs.astype(np.float64) + 0.5) * dz,
        ], axis=1)
        quad = [base + np.array([ox * half[0], oy * half[1], oz * half[2]]) for ox, oy, oz in corners]
        face_list.append(np.stack(quad, axis=1))
        color = OCC_COLORS[occ[xs, ys, zs]][:, ::-1]
        if shade != 1.0:
            color = (color.astype(np.float32) * shade).astype(np.uint8)
        color_list.append(color)
    if face_list:
        faces = np.concatenate(face_list, axis=0)
        colors = np.concatenate(color_list, axis=0)
        uv, depth, valid = _project(faces.reshape(-1, 3), eye, rot)
        uv = to_pixel(uv).reshape(-1, 4, 2)
        depth = depth.reshape(-1, 4)
        valid = valid.reshape(-1, 4).all(axis=1)
        for face_id in np.argsort(-depth.mean(axis=1)):
            if not valid[face_id]:
                continue
            poly = np.round(uv[face_id]).astype(np.int32)
            cv2.fillConvexPoly(canvas, poly, colors[face_id].tolist(), cv2.LINE_AA)
    _draw_persp_axes(canvas, eye, rot, to_pixel)
    return canvas


def panel(image, title):
    bar = np.full((28, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin'):
        import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.ann_file = cfg.data.train.ann_file
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    tasks = cfg.model.tasks
    if isinstance(tasks, str):
        tasks = ('det', 'occ') if tasks == 'both' else (tasks,)
    show_det = 'det' in tasks
    show_occ = 'occ' in tasks
    pc = cfg.point_cloud_range
    det_bounds = (pc[0], pc[1], pc[3], pc[4])
    label_range = list(cfg.model.occ_head.get(
        'occ_label_range', [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]))
    z_bounds = (label_range[2], label_range[5])
    grid = cfg.model.img_view_transformer.grid_config
    class_names = list(cfg.class_names)
    occ_index = _index_occ_root(cfg.occ_root) if show_occ else {}
    os.makedirs(args.out_dir, exist_ok=True)
    limit = len(dataset) if args.max_frames <= 0 else min(args.max_frames, len(dataset))

    for index, data in enumerate(data_loader):
        if index >= limit:
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        info = dataset.data_infos[index]
        lidar2ego = lidar_to_ego_matrix(info)
        ann = dataset.get_ann_info(index)
        gt_boxes = ann['gt_bboxes_3d'] if show_det else None
        gt_labels = ann['gt_labels_3d'] if show_det else None
        pred_boxes = pred_labels = pred_scores = None
        if show_det and result.get('pts_bbox') is not None:
            pred = result['pts_bbox']
            keep = pred['scores_3d'] >= args.score_thr
            pred_boxes = pred['boxes_3d'][keep]
            pred_labels = pred['labels_3d'][keep]
            pred_scores = pred['scores_3d'][keep]

        rows = []
        for names in CAM_LAYOUT:
            cams = []
            for name in names:
                cam = info['cams'][name]
                image = cv2.imread(cam['data_path'])
                src_hw = image.shape[:2]
                image = cv2.resize(image, (IMG_W, IMG_H))
                if show_det:
                    lidar2img = scale_lidar2img(
                        camera_ego2img(cam, lidar2ego), src_hw, (IMG_H, IMG_W))
                    draw_boxes_on_image(
                        image, gt_boxes, gt_labels, lidar2img, class_names, color=(0, 180, 0))
                    draw_boxes_on_image(
                        image, pred_boxes, pred_labels, lidar2img, class_names, scores=pred_scores)
                cams.append(panel(image, name))
            rows.append(np.concatenate(cams, axis=1))

        bottom = []
        raised = []
        pts = None
        if show_det:
            canvas = np.full((PLAN_SIZE, PLAN_SIZE, 3), 255, dtype=np.uint8)
            pts = transform_points_lidar_to_ego(
                np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5),
                lidar2ego)
            draw_points(canvas, pts, det_bounds)
            draw_boxes(canvas, gt_boxes, gt_labels, det_bounds, color=(0, 180, 0))
            draw_boxes(canvas, pred_boxes, pred_labels, det_bounds, scores=pred_scores)
            draw_axes(canvas, det_bounds)
            bottom.append(panel(canvas, 'det  green=gt  color=pred'))
            raised.append(panel(render_det_3d(
                pts, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                det_bounds, z_bounds), 'det 3d'))
        if show_occ:
            gt_path = occ_index.get(info['token'])
            if gt_path is None:
                raise FileNotFoundError('没有找到 {} 的占用标签'.format(info['token']))
            gt = np.load(gt_path)['semantics']
            pred_occ, pred_bounds = crop_occ(result['occ'], grid, label_range)
            gt_bounds = (label_range[0], label_range[1], label_range[3], label_range[4])
            bottom.append(panel(colorize_occ(gt, PLAN_SIZE, gt_bounds), 'occ gt'))
            bottom.append(panel(colorize_occ(pred_occ, PLAN_SIZE, pred_bounds), 'occ pred'))
            raised.append(panel(render_occ_3d(gt, gt_bounds, z_bounds), 'occ gt 3d'))
            raised.append(panel(render_occ_3d(pred_occ, pred_bounds, z_bounds), 'occ pred 3d'))
        if bottom:
            rows.append(np.concatenate(bottom, axis=1))
        if raised:
            rows.append(np.concatenate(raised, axis=1))
        image = stitch_rows(rows)
        name = '{:03d}_{}.jpg'.format(index, info['token'][:8])
        cv2.imwrite(osp.join(args.out_dir, name), image)
        print(index, name)
    print('saved to', args.out_dir)


if __name__ == '__main__':
    main()
