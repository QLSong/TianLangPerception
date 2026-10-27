"""用 det10m 权重跑自有相机和激光雷达。

数据和标定默认读配置里的 demo_data_root、demo_calib。点云先变到自车（x 朝前、y 朝左），
再按验证集同一套缩放裁剪把图像收成 256×704。检测框和占用都画在这个坐标系里。

示例：
/opt/streampetr/bin/python tools/demo_det10m.py \
  projects/configs/TianLangEyes/streampetr_convnext_pp_det10m_demo.py \
  work_dirs/streampetr_convnext_pp_det10m_nus/latest.pth
"""
import argparse
import json
import os
import os.path as osp
import re
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import cv2
import mmcv
import numpy as np
import torch
import yaml
from mmcv import Config
from mmdet3d.models import build_model
from nuscenes.eval.common.utils import Quaternion

import projects.mmdet3d_plugin  # noqa: F401
import vis_mini
from projects.mmdet3d_plugin.core.checkpoint import load_pretrained
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
    invert_matrix_egopose_numpy, transform_points_lidar_to_ego)
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipRotImage

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
LIDAR_EXT = ('.bin', '.pcd', '.npy')
LIDAR_DIR_NAMES = ('lidar', 'lidar_top', 'rslidar', 'pointcloud', 'points', 'pandar', 'velodyne')
CAM_ALIASES = {
    'CAM_FRONT': ('cam_front', 'front', 'front_wide', 'camera_front', 'front_camera'),
    'CAM_FRONT_RIGHT': ('cam_front_right', 'front_right', 'camera_front_right', 'right_front'),
    'CAM_FRONT_LEFT': ('cam_front_left', 'front_left', 'camera_front_left', 'left_front'),
    'CAM_BACK': ('cam_back', 'back', 'rear', 'camera_back', 'camera_rear', 'back_camera'),
    'CAM_BACK_LEFT': ('cam_back_left', 'back_left', 'rear_left', 'left_back'),
    'CAM_BACK_RIGHT': ('cam_back_right', 'back_right', 'rear_right', 'right_back'),
}
POSE_TO_CAM = ('ego2cam', 'ego_to_cam', 'T_cam_ego', 'T_ego_to_cam')
POSE_FROM_CAM = (
    'cam2ego', 'cam_to_ego', 'sensor2ego', 'T_ego_cam', 'T_cam_to_ego',
    'extrinsic', 'camera_extrinsic')
INTRINSIC_KEYS = (
    'intrinsic', 'intrinsics', 'K', 'camera_matrix', 'camera_K', 'camera_intrinsic')
DISTORTION_KEYS = ('distortion', 'dist', 'dist_coeffs', 'D', 'distortion_coeffs')
FISHEYE_MODELS = ('fisheye', 'equidistant', 'kannala-brandt', 'kannala_brandt')


def parse_args():
    parser = argparse.ArgumentParser(description='det10m 自有数据推理')
    parser.add_argument('config')
    parser.add_argument('checkpoint', nargs='?', default=None)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--calib', default=None)
    parser.add_argument('--out-dir', default='work_dirs/demo_det10m')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--max-frames', type=int, default=0, help='0 表示全部')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def _norm_name(name):
    return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')


def _alias_table(camera_names):
    table = {}
    for name in camera_names:
        table[_norm_name(name)] = name
        for alias in CAM_ALIASES.get(name, ()):
            table[_norm_name(alias)] = name
    return table


def _load_yaml(path):
    text = open(path, 'r', encoding='utf-8').read().replace('\r\n', '\n')
    if text.lstrip().startswith('%YAML') or '!!opencv-matrix' in text:
        pattern = re.compile(
            r'!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[(.*?)\]',
            re.S)
        def repl(match):
            rows, cols = int(match.group(1)), int(match.group(2))
            nums = [float(item) for item in re.findall(
                r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', match.group(3))]
            return json.dumps(np.array(nums, dtype=np.float64).reshape(rows, cols).tolist())
        text = pattern.sub(repl, text)
        text = re.sub(r'^%YAML:.*\n(?:---\n)?', '', text)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RuntimeError('标定 {} 解析失败: {}'.format(path, exc)) from exc
    if not isinstance(data, dict):
        raise RuntimeError('标定 {} 不是字典'.format(path))
    return data


def _as_floats(value):
    if isinstance(value, dict):
        if 'data' in value and 'rows' in value and 'cols' in value:
            rows, cols = int(value['rows']), int(value['cols'])
            return np.array(value['data'], dtype=np.float64).reshape(rows, cols)
        if 'rotation' in value or 'quaternion' in value:
            rotation = value.get('rotation', value.get('quaternion'))
            translation = value.get('translation', value.get('t', value.get('position')))
            quat = np.asarray(rotation, dtype=np.float64).reshape(-1)
            if quat.size == 9:
                rot = quat.reshape(3, 3)
            elif quat.size == 4:
                if str(value.get('order', 'wxyz')).lower() == 'xyzw':
                    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
                rot = Quaternion(quat).rotation_matrix
            else:
                raise RuntimeError('旋转维度是 {}'.format(quat.size))
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = rot
            mat[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
            return mat
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3, 3):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = array
        return mat
    if array.size == 9:
        return _as_floats(array.reshape(3, 3))
    if array.shape == (3, 4):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = array
        return mat
    if array.shape == (4, 4) or array.size == 16:
        return array.reshape(4, 4)
    if array.size == 4:
        return array.reshape(4)
    return array


def _first(node, keys):
    for key in keys:
        if isinstance(node, dict) and key in node and node[key] is not None:
            return node[key]
    return None


def _intrinsic_matrix(node):
    raw = _first(node, INTRINSIC_KEYS)
    if raw is None and all(key in node for key in ('fx', 'fy', 'cx', 'cy')):
        raw = [node['fx'], node['fy'], node['cx'], node['cy']]
    if raw is None:
        raise KeyError('缺少内参')
    if isinstance(raw, dict) and 'data' in raw:
        raw = _as_floats(raw)
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 4:
        fx, fy, cx, cy = array.reshape(4)
        array = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    if array.shape != (3, 3):
        array = array.reshape(3, 3)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = array
    return mat


def _ego2cam(node, lidar2ego):
    raw = _first(node, POSE_TO_CAM)
    if raw is not None:
        return np.asarray(_as_floats(raw), dtype=np.float32)
    raw = _first(node, POSE_FROM_CAM)
    if raw is not None:
        return invert_matrix_egopose_numpy(np.asarray(_as_floats(raw), dtype=np.float32))
    raw = _first(node, ('lidar2cam', 'lidar_to_cam', 'T_cam_lidar'))
    if raw is not None:
        lidar2cam = np.asarray(_as_floats(raw), dtype=np.float32)
        return lidar2cam @ invert_matrix_egopose_numpy(lidar2ego)
    raw = _first(node, ('cam2lidar', 'cam_to_lidar', 'sensor2lidar', 'T_lidar_cam'))
    if raw is not None:
        lidar2cam = invert_matrix_egopose_numpy(np.asarray(_as_floats(raw), dtype=np.float32))
        return lidar2cam @ invert_matrix_egopose_numpy(lidar2ego)
    rotation = _first(node, ('R', 'rotation_matrix'))
    translation = _first(node, ('t', 'translation'))
    if rotation is not None and translation is not None:
        # 没有标明方向时，R、t 视为相机在自车下的位姿，即相机到自车。
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
        pose[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
        return invert_matrix_egopose_numpy(pose)
    raise KeyError('缺少外参')


def _camera_nodes(calib):
    for key in ('cameras', 'camera', 'cams'):
        if key in calib:
            node = calib[key]
            if isinstance(node, dict):
                return list(node.items())
            if isinstance(node, list):
                pairs = []
                for item in node:
                    name = item.get('name', item.get('frame_id', item.get('id')))
                    pairs.append((name, item))
                return pairs
    pairs = []
    for key, value in calib.items():
        if isinstance(value, dict) and (_first(value, INTRINSIC_KEYS) is not None or 'fx' in value):
            pairs.append((key, value))
    return pairs


def _lidar2ego(calib):
    for key in ('lidar2ego', 'lidar_to_ego', 'T_ego_lidar'):
        if key in calib:
            return np.asarray(_as_floats(calib[key]), dtype=np.float32)
    lidar = calib.get('lidar', calib.get('LIDAR', calib.get('LIDAR_TOP')))
    if isinstance(lidar, dict):
        raw = _first(lidar, ('lidar2ego', 'lidar_to_ego', 'T_ego_lidar', 'extrinsic'))
        if raw is not None:
            return np.asarray(_as_floats(raw), dtype=np.float32)
        raw = _first(lidar, ('ego2lidar', 'T_lidar_ego'))
        if raw is not None:
            return invert_matrix_egopose_numpy(np.asarray(_as_floats(raw), dtype=np.float32))
    return np.eye(4, dtype=np.float32)


def load_calibration(path, camera_names):
    calib = _load_yaml(path)
    if not _camera_nodes(calib):
        for key in ('calibration', 'calib', 'sensors'):
            if isinstance(calib.get(key), dict):
                calib = calib[key]
                break
    aliases = _alias_table(camera_names)
    lidar2ego = _lidar2ego(calib)
    found = {}
    for name, node in _camera_nodes(calib):
        canonical = aliases.get(_norm_name(name))
        if canonical is None or not isinstance(node, dict):
            continue
        distortion = _first(node, DISTORTION_KEYS)
        if distortion is not None:
            distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
        model = node.get('distortion_model', node.get('model', ''))
        found[canonical] = dict(
            intrinsic=_intrinsic_matrix(node),
            ego2cam=_ego2cam(node, lidar2ego),
            distortion=distortion,
            distortion_model=str(model or ''))
    missing = [name for name in camera_names if name not in found]
    if missing:
        known = [name for name, _ in _camera_nodes(calib)]
        raise RuntimeError(
            '标定里缺少 {}。读到的相机名是 {}。文件顶层键是 {}'.format(
                missing, known, list(calib.keys())))
    return lidar2ego, [found[name] for name in camera_names]


def _list_files(folder, suffixes):
    files = []
    for name in sorted(os.listdir(folder)):
        path = osp.join(folder, name)
        if osp.isfile(path) and name.lower().endswith(suffixes):
            files.append(path)
    return files


def _stem_key(path):
    stem = osp.splitext(osp.basename(path))[0]
    nums = re.findall(r'\d+', stem)
    return nums[-1] if nums else stem


def _match_dirs(root, camera_names):
    aliases = _alias_table(camera_names)
    found = {}
    lidar_dir = None
    for dirpath, dirnames, _ in os.walk(root):
        for name in list(dirnames):
            canonical = aliases.get(_norm_name(name))
            if canonical and canonical not in found:
                found[canonical] = osp.join(dirpath, name)
            if _norm_name(name) in LIDAR_DIR_NAMES and lidar_dir is None:
                lidar_dir = osp.join(dirpath, name)
        if len(found) == len(camera_names) and lidar_dir:
            break
    return found, lidar_dir


def _frames_from_dirs(cam_dirs, lidar_dir, camera_names):
    groups = []
    for name in camera_names:
        files = _list_files(cam_dirs[name], IMG_EXT)
        if not files:
            raise FileNotFoundError('{} 里没有图像'.format(cam_dirs[name]))
        groups.append({_stem_key(path): path for path in files})
    lidar_files = _list_files(lidar_dir, LIDAR_EXT)
    if not lidar_files:
        raise FileNotFoundError('{} 里没有点云'.format(lidar_dir))
    lidar = {_stem_key(path): path for path in lidar_files}
    keys = set(lidar)
    for group in groups:
        keys &= set(group)
    if not keys:
        raise RuntimeError('六路图像和点云没有共同的时间戳文件名')
    frames = []
    for key in sorted(keys):
        frames.append(dict(
            key=key,
            images=[group[key] for group in groups],
            lidar=lidar[key]))
    return frames


def _frames_from_samples(root, camera_names):
    aliases = _alias_table(camera_names)
    frames = []
    for name in sorted(os.listdir(root)):
        folder = osp.join(root, name)
        if not osp.isdir(folder):
            continue
        images = {}
        lidar = None
        for file_name in os.listdir(folder):
            path = osp.join(folder, file_name)
            if not osp.isfile(path):
                continue
            stem, ext = osp.splitext(file_name)
            if ext.lower() in IMG_EXT:
                canonical = aliases.get(_norm_name(stem))
                if canonical:
                    images[canonical] = path
            elif ext.lower() in LIDAR_EXT and lidar is None:
                lidar = path
        if lidar and all(cam in images for cam in camera_names):
            frames.append(dict(
                key=name,
                images=[images[cam] for cam in camera_names],
                lidar=lidar))
    return frames


def index_frames(root, camera_names):
    if not osp.isdir(root):
        raise FileNotFoundError('数据目录不存在: {}'.format(root))
    frames = _frames_from_samples(root, camera_names)
    if frames:
        return frames
    cam_dirs, lidar_dir = _match_dirs(root, camera_names)
    missing = [name for name in camera_names if name not in cam_dirs]
    if missing or lidar_dir is None:
        raise RuntimeError(
            '没有在 {} 里找到六路相机目录和雷达目录。相机缺 {}，雷达目录是 {}。'.format(
                root, missing, lidar_dir))
    return _frames_from_dirs(cam_dirs, lidar_dir, camera_names)


def _pcd_dtype(type_name, size):
    table = {
        ('F', 4): np.float32, ('F', 8): np.float64,
        ('U', 1): np.uint8, ('U', 2): np.uint16, ('U', 4): np.uint32,
        ('I', 1): np.int8, ('I', 2): np.int16, ('I', 4): np.int32,
    }
    dtype = table.get((type_name, size))
    if dtype is None:
        raise RuntimeError('不支持的 PCD 字段 {}{}'.format(type_name, size))
    return dtype


def _read_pcd(path):
    with open(path, 'rb') as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError('点云 {} 没有 DATA 行'.format(path))
            header.append(line.decode('ascii', errors='ignore').strip())
            if header[-1].upper().startswith('DATA'):
                payload = handle.read()
                break
    fields = {}
    for line in header:
        parts = line.split()
        if parts:
            fields[parts[0].upper()] = parts[1:]
    if 'POINTS' in fields:
        count = int(fields['POINTS'][0])
    else:
        count = int(fields['WIDTH'][0]) * int(fields['HEIGHT'][0])
    names = [item.lower() for item in fields.get('FIELDS', ['x', 'y', 'z'])]
    sizes = [int(item) for item in fields.get('SIZE', ['4'] * len(names))]
    types = [item.upper() for item in fields.get('TYPE', ['F'] * len(names))]
    counts = [int(item) for item in fields.get('COUNT', ['1'] * len(names))]
    if any(item != 1 for item in counts):
        raise RuntimeError('不支持多元素 PCD 字段 {}'.format(path))
    data_type = fields['DATA'][0].lower()
    if data_type == 'ascii':
        text = payload.decode('ascii', errors='ignore').strip().splitlines()
        raw = np.loadtxt(text[:count], dtype=np.float32)
    elif data_type == 'binary':
        stride = sum(sizes)
        buf = np.frombuffer(payload, dtype=np.uint8, count=count * stride)
        if buf.size != count * stride:
            raise RuntimeError('点云 {} 的二进制长度不够'.format(path))
        raw = np.zeros((count, len(names)), dtype=np.float32)
        offset = 0
        for col, (size, type_name) in enumerate(zip(sizes, types)):
            piece = buf.reshape(count, stride)[:, offset:offset + size]
            values = np.ascontiguousarray(piece).view(_pcd_dtype(type_name, size)).reshape(count)
            raw[:, col] = values.astype(np.float32)
            offset += size
    else:
        raise RuntimeError('不支持压缩 PCD {}'.format(path))
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    index = [names.index(axis) for axis in ('x', 'y', 'z')]
    xyz = raw[:, index]
    if 'intensity' in names:
        intensity = raw[:, names.index('intensity'):names.index('intensity') + 1]
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    return np.concatenate([xyz, intensity], axis=1).astype(np.float32)


def read_points(path):
    if path.endswith('.npy'):
        points = np.load(path).astype(np.float32)
    elif path.endswith('.pcd'):
        points = _read_pcd(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        for width in (5, 4, 6, 3):
            if raw.size % width == 0:
                points = raw.reshape(-1, width)
                break
        else:
            raise RuntimeError('点云 {} 的长度 {} 不能切开'.format(path, raw.size))
    if points.shape[1] == 3:
        points = np.concatenate([points, np.zeros((points.shape[0], 1), np.float32)], axis=1)
    points = points[:, :4]
    keep = np.isfinite(points).all(axis=1)
    return points[keep]


def undistort(image, intrinsic, distortion, model=''):
    if distortion is None or distortion.size == 0 or np.allclose(distortion, 0):
        return image
    camera = np.asarray(intrinsic[:3, :3], dtype=np.float64)
    name = str(model or '').lower()
    fisheye = name in FISHEYE_MODELS or (not name and distortion.size == 4)
    if fisheye:
        coeffs = np.zeros((4, 1), dtype=np.float64)
        coeffs[:min(4, distortion.size), 0] = distortion[:4]
        return cv2.fisheye.undistortImage(image, camera, coeffs)
    return cv2.undistort(image, camera, distortion.astype(np.float64))


def prepare_images(frame, cameras, final_dim, mean, std):
    images = []
    intrinsics = []
    extrinsics = []
    lidar2img = []
    frame['vis_images'] = []
    warp = ResizeCropFlipRotImage(dict(
        resize_lim=(1.0, 1.0), final_dim=final_dim, bot_pct_lim=(0.0, 0.0),
        rot_lim=(0.0, 0.0), H=1, W=1, rand_flip=False), training=False, with_2d=False)
    for path, camera in zip(frame['images'], cameras):
        image = mmcv.imread(path, 'unchanged')
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = undistort(
            image, camera['intrinsic'], camera['distortion'], camera['distortion_model'])
        height, width = image.shape[:2]
        warp.data_aug_conf['H'] = height
        warp.data_aug_conf['W'] = width
        packed = dict(
            img=[image],
            intrinsics=[camera['intrinsic'].copy()],
            extrinsics=[camera['ego2cam'].copy()])
        warp(packed)
        normed = mmcv.imnormalize(packed['img'][0], mean, std, to_rgb=True)
        images.append(np.transpose(normed, (2, 0, 1)))
        intrinsics.append(np.asarray(packed['intrinsics'][0], dtype=np.float32))
        extrinsics.append(np.asarray(packed['extrinsics'][0], dtype=np.float32))
        lidar2img.append(np.asarray(packed['lidar2img'][0], dtype=np.float32))
        frame['vis_images'].append(np.asarray(packed['img'][0], dtype=np.uint8))
    return images, intrinsics, extrinsics, lidar2img


def draw_occ_bev(canvas, occ, bounds):
    occupied = occ != vis_mini.EMPTY
    if not np.any(occupied):
        return
    top = occupied.shape[2] - 1 - np.argmax(occupied[:, :, ::-1], axis=2)
    top = np.where(occupied.any(axis=2), top, -1)
    height, width = canvas.shape[:2]
    xs, ys = np.nonzero(top >= 0)
    labels = occ[xs, ys, top[xs, ys]]
    centers_x = bounds[0] + (xs.astype(np.float32) + 0.5) * (bounds[2] - bounds[0]) / occ.shape[0]
    centers_y = bounds[1] + (ys.astype(np.float32) + 0.5) * (bounds[3] - bounds[1]) / occ.shape[1]
    pix = vis_mini.world_to_pixel(centers_x, centers_y, bounds, width, height).astype(np.int32)
    for pixel, label in zip(pix, labels):
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            continue
        color = vis_mini.OCC_COLORS[int(label) % len(vis_mini.OCC_COLORS)]
        cv2.circle(
            canvas, (int(pixel[0]), int(pixel[1])), 2,
            (int(color[2]), int(color[1]), int(color[0])), -1, cv2.LINE_AA)


def render(frame, boxes, labels, scores, occ, lidar2img, bounds, class_names, z_bounds, camera_names):
    tiles = []
    for image, mat in zip(frame['vis_images'], lidar2img):
        view = image.copy()
        vis_mini.draw_boxes_on_image(view, boxes, labels, mat, class_names, scores=scores)
        if occ is not None:
            vis_mini.draw_occ_on_image(view, occ, bounds, z_bounds, mat)
        tiles.append(cv2.resize(view, (vis_mini.IMG_W, vis_mini.IMG_H)))
    order = [name for row in vis_mini.CAM_LAYOUT for name in row]
    tiles = [tiles[camera_names.index(name)] for name in order]
    top = np.concatenate(tiles[:3], axis=1)
    bottom = np.concatenate(tiles[3:], axis=1)
    views = np.concatenate([top, bottom], axis=0)
    bev = np.zeros((views.shape[0], vis_mini.PLAN_SIZE, 3), dtype=np.uint8)
    vis_mini.draw_axes(bev, bounds)
    if frame.get('points') is not None:
        vis_mini.draw_points(bev, frame['points'], bounds)
    if occ is not None:
        draw_occ_bev(bev, occ, bounds)
    vis_mini.draw_boxes(bev, boxes, labels, bounds, scores=scores)
    return np.concatenate([views, bev], axis=1)


def build_model_from_cfg(cfg, checkpoint, device):
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_pretrained(model, checkpoint)
    model.to(device)
    model.eval()
    return model


def run_frame(model, frame, cameras, cfg, device):
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    final_dim = tuple(cfg.ida_aug_conf['final_dim'])
    images, intrinsics, extrinsics, mats = prepare_images(frame, cameras, final_dim, mean, std)
    points = transform_points_lidar_to_ego(read_points(frame['lidar']), frame['lidar2ego'])
    frame['points'] = points
    image = torch.from_numpy(np.stack(images, axis=0)).unsqueeze(0).to(device)
    data = dict(
        img=image,
        points=[torch.from_numpy(points)],
        lidar2img=torch.from_numpy(np.stack(mats)).unsqueeze(0).to(device),
        intrinsics=torch.from_numpy(np.stack(intrinsics)).unsqueeze(0).to(device),
        extrinsics=torch.from_numpy(np.stack(extrinsics)).unsqueeze(0).to(device),
    )
    height, width = images[0].shape[-2:]
    meta = dict(
        scene_token='demo',
        pad_shape=[(height, width, 3)] * len(cameras),
        img_shape=[(height, width, 3)] * len(cameras),
    )
    with torch.no_grad():
        result = model.simple_test([meta], **data)[0]
    pred = result.get('pts_bbox') or {}
    return (
        pred.get('boxes_3d'), pred.get('labels_3d'), pred.get('scores_3d'),
        result.get('occ'), mats)


def filter_boxes(boxes, labels, scores, score_thr):
    if boxes is None or scores is None or len(scores) == 0:
        return boxes, labels, scores
    scores = scores.detach().cpu()
    labels = labels.detach().cpu()
    boxes = boxes.to('cpu')
    keep = scores >= score_thr
    return boxes[keep], labels[keep], scores[keep]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin'):
        __import__(cfg.plugin_dir.strip('/').replace('/', '.'))
    checkpoint = args.checkpoint or cfg.load_from
    data_root = args.data_root or cfg.demo_data_root
    calib_path = args.calib or cfg.demo_calib
    camera_names = list(cfg.demo_cameras)
    lidar2ego, cameras = load_calibration(calib_path, camera_names)
    frames = index_frames(data_root, camera_names)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    print('标定 {}，雷达到自车平移 {}'.format(calib_path, lidar2ego[:3, 3].round(3).tolist()))
    for name, camera in zip(camera_names, cameras):
        position = invert_matrix_egopose_numpy(camera['ego2cam'])[:3, 3]
        print('{} 在自车下的位置 {}'.format(name, np.round(position, 3).tolist()))
    print('共 {} 帧，输出到 {}'.format(len(frames), args.out_dir))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model_from_cfg(cfg, checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)
    occ_range = [float(v) for v in cfg.occ_range]
    bounds = (occ_range[0], occ_range[1], occ_range[3], occ_range[4])
    z_bounds = (occ_range[2], occ_range[5])
    class_names = list(cfg.class_names)
    records = []
    for index, frame in enumerate(frames):
        frame['lidar2ego'] = lidar2ego
        boxes, labels, scores, occ, mats = run_frame(model, frame, cameras, cfg, device)
        boxes, labels, scores = filter_boxes(boxes, labels, scores, args.score_thr)
        image = render(
            frame, boxes, labels, scores, occ, mats, bounds, class_names, z_bounds, camera_names)
        name = '{:04d}_{}.jpg'.format(index, frame['key'])
        cv2.imwrite(osp.join(args.out_dir, name), image)
        count = 0 if labels is None else int(len(labels))
        records.append(dict(frame=frame['key'], boxes=count))
        print('{} 检测框 {}'.format(name, count))
    mmcv.dump(records, osp.join(args.out_dir, 'frames.json'))


if __name__ == '__main__':
    main()
