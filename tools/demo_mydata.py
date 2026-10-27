"""用 det10m 权重跑 test_my_data。

数据目录下按传感器分文件夹：cam0 到 cam3 是 jpg，front_up 和 back_down 是 pcd。
pcd 已在 base_link。旧的 test_data/*.bin 仍按雷达轴乘外参。

图像保持畸变送进网络。IPM 用内参、外参和 rational 畸变把特征投到自车 BEV。
两路雷达乘各自到 base_link 的外参，再按 x、y 正投影到 BEV。

示例：
/opt/streampetr/bin/python tools/demo_mydata.py \
  projects/configs/TianLangEyes/streampetr_convnext_pp_det10m_mydata.py \
  work_dirs/streampetr_convnext_pp_det10m_nus/latest.pth --max-frames 20
"""
import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import cv2
import mmcv
import numpy as np
import torch
import yaml
from mmcv import Config
from mmdet3d.core.bbox import get_box_type
from mmdet3d.models import build_model

import projects.mmdet3d_plugin  # noqa: F401
import vis_mini
from projects.mmdet3d_plugin.core.checkpoint import load_pretrained
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import ResizeCropFlipRotImage

POINT_STRIDE = 27
LIDAR_DIRS = {
    'front_up': 'front_lidar',
    'back_down': 'rear_lidar',
}
# Occ3D 的地面：可行驶、其他平面、人行道、地形。
GROUND_LABELS = np.array([11, 12, 13, 14], dtype=np.int64)
SHOW_OCC = False


def parse_args():
    parser = argparse.ArgumentParser(description='test_my_data 推理')
    parser.add_argument('config')
    parser.add_argument('checkpoint', nargs='?', default=None)
    parser.add_argument('--data-root', default='/mnt/data/nas/datasets/test_my_data/2026-10-26-09-21-26')
    parser.add_argument('--calib', default=None)
    parser.add_argument('--out-dir', default='work_dirs/demo_mydata')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--max-frames', type=int, default=0, help='最多渲染多少帧，0 表示不限制')
    parser.add_argument('--stride', type=int, default=10, help='每隔多少帧渲染一帧')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    sensors = data.get('sensors', data)
    if not isinstance(sensors, dict):
        raise RuntimeError('标定 {} 里没有 sensors'.format(path))
    return sensors


def camera_from_calib(node):
    intrinsic = np.eye(4, dtype=np.float32)
    intrinsic[:3, :3] = np.asarray(node['K'], dtype=np.float32)
    distortion = np.zeros(8, dtype=np.float32)
    coeffs = np.asarray(node['D'], dtype=np.float32).reshape(-1)
    if coeffs.size > 8:
        raise RuntimeError('畸变系数有 {} 个'.format(coeffs.size))
    distortion[:coeffs.size] = coeffs
    return dict(
        intrinsic=intrinsic,
        ego2cam=np.asarray(node['T_sensor_base'], dtype=np.float32),
        distortion=distortion,
        model=str(node.get('camera_model', '')),
        size=tuple(int(v) for v in node['image_size']))


def load_sensors(path, camera_names):
    sensors = load_yaml(path)
    cameras = []
    for name in camera_names:
        if name not in sensors:
            raise RuntimeError('标定缺少相机 {}。已有 {}'.format(name, list(sensors)))
        cameras.append(camera_from_calib(sensors[name]))
    lidars = {}
    for folder, key in LIDAR_DIRS.items():
        if key not in sensors:
            raise RuntimeError('标定缺少雷达 {}'.format(key))
        lidars[folder] = np.asarray(sensors[key]['T_base_sensor'], dtype=np.float32)
    return cameras, lidars


def frame_key(file_name):
    """000000.jpg 和 000264_1792571546.700.bin 都取帧号 000000 / 000264。"""
    return osp.splitext(file_name)[0].split('_', 1)[0]


def index_frames(root, camera_names):
    folder = osp.join(root, 'test_data')
    if not osp.isdir(folder):
        folder = root
    groups = {}
    for name in list(camera_names) + list(LIDAR_DIRS):
        path = osp.join(folder, name)
        if not osp.isdir(path):
            raise FileNotFoundError('缺少 {}'.format(path))
        groups[name] = {}
        for file_name in os.listdir(path):
            if file_name.startswith('.'):
                continue
            groups[name][frame_key(file_name)] = osp.join(path, file_name)
    keys = set.intersection(*(set(item) for item in groups.values()))
    if not keys:
        raise RuntimeError('相机和两路雷达没有共同帧号')
    frames = []
    for key in sorted(keys):
        frames.append(dict(
            key=key,
            images=[groups[name][key] for name in camera_names],
            lidars={name: groups[name][key] for name in LIDAR_DIRS}))
    return frames


def read_pcd(path):
    """binary PCD，字段是 float32 的 x y z intensity。"""
    with open(path, 'rb') as handle:
        fields = {}
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError('{} 没有 DATA 行'.format(path))
            parts = line.decode('ascii', errors='ignore').strip().split()
            if not parts:
                continue
            fields[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                payload = handle.read()
                break
    names = [item.lower() for item in fields['FIELDS']]
    sizes = [int(item) for item in fields.get('SIZE', ['4'] * len(names))]
    types = [item.upper() for item in fields.get('TYPE', ['F'] * len(names))]
    if any(size != 4 or type_name != 'F' for size, type_name in zip(sizes, types)):
        raise RuntimeError('{} 的字段不是 float32'.format(path))
    if fields['DATA'][0].lower() != 'binary':
        raise RuntimeError('{} 不是 binary pcd'.format(path))
    count = int(fields['POINTS'][0]) if 'POINTS' in fields else int(fields['WIDTH'][0])
    raw = np.frombuffer(payload, dtype=np.float32, count=count * len(names)).reshape(count, len(names))
    xyz = raw[:, [names.index(axis) for axis in ('x', 'y', 'z')]]
    if 'intensity' in names:
        intensity = raw[:, names.index('intensity'):names.index('intensity') + 1]
    else:
        intensity = np.zeros((count, 1), np.float32)
    points = np.concatenate([xyz, intensity.astype(np.float32)], axis=1)
    return points[np.isfinite(points).all(axis=1)]


def read_lidar(path):
    if path.endswith('.pcd'):
        return read_pcd(path)
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % POINT_STRIDE != 0:
        raise RuntimeError('{} 的长度 {} 不是 {} 的倍数'.format(path, raw.size, POINT_STRIDE))
    rec = raw.reshape(-1, POINT_STRIDE)
    xyz = rec[:, :12].copy().view(np.float32).reshape(-1, 3)
    intensity = rec[:, 12:16].copy().view(np.float32).reshape(-1, 1)
    points = np.concatenate([xyz, intensity], axis=1)
    return points[np.isfinite(points).all(axis=1)]


# 两路 rslidar 的点云轴和标定传感器轴不是同一套。直接乘 T_base_sensor 时地面会立起来，
# 俯视被 z∈[-1, 1] 切成一条。下面的矩阵把点云变到标定轴：p_calib = R @ p_cloud。
# 前雷达 (x, y, z) -> (y, z, x)，后雷达 (x, y, z) -> (-y, -x, -z)。
# 地面都落在 z≈-0.5。后雷达变到 base 之后再对调 x、y，走廊才沿 -x，而不是横在 y 上。
CLOUD_TO_CALIB = {
    'front_up': np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32),
    'back_down': np.array([
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32),
}


def cloud_to_base(name, sensor_to_base):
    """点云文件到 base_link 的外参。标定 T_base_sensor 用的是传感器轴，文件轴先换过去。"""
    extrinsic = np.asarray(sensor_to_base, dtype=np.float32).copy()
    align = np.eye(4, dtype=np.float32)
    align[:3, :3] = CLOUD_TO_CALIB[name]
    return extrinsic @ align


def to_base(points, extrinsic):
    """p_base = T_base_cloud @ p_cloud。列向量，外参是 4×4。"""
    out = np.array(points, dtype=np.float32, copy=True)
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
    out[:, :3] = out[:, :3] @ rotation.T + translation
    return out


def merge_lidars(frame, lidar_poses):
    clouds = []
    for name, transform in lidar_poses.items():
        path = frame['lidars'][name]
        cloud = read_lidar(path)
        # 这条记录的 pcd 已经在 base_link，地面在 z≈-0.5。旧 bin 仍是雷达轴，要乘外参。
        if not path.endswith('.pcd'):
            cloud = to_base(cloud, cloud_to_base(name, transform))
            if name == 'back_down':
                cloud[:, [0, 1]] = cloud[:, [1, 0]]
        clouds.append(cloud)
    return np.concatenate(clouds, axis=0)


def paint_lidar_bev(canvas, points, bounds, z_bounds):
    """外参已经变到 base。BEV 是 x、y 正投影，高度留在占用范围并包含站立高度。"""
    if points is None or len(points) == 0:
        return
    z0, z1 = float(z_bounds[0]), max(float(z_bounds[1]), 2.0)
    keep = (
        (points[:, 0] >= bounds[0]) & (points[:, 0] < bounds[2])
        & (points[:, 1] >= bounds[1]) & (points[:, 1] < bounds[3])
        & (points[:, 2] >= z0) & (points[:, 2] <= z1))
    xy = points[keep][:, :2]
    if xy.shape[0] == 0:
        return
    height, width = canvas.shape[:2]
    pix = vis_mini.world_to_pixel(xy[:, 0], xy[:, 1], bounds, width, height).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, width - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, height - 1)
    canvas[pix[:, 1], pix[:, 0]] = (220, 220, 220)


def prepare_images(frame, cameras, final_dim, mean, std):
    """畸变图像缩放到网络输入，不去畸变。内参乘上同一次缩放和平移，IPM 用这份内参。"""
    images, intrinsics, extrinsics, mats, distortion = [], [], [], [], []
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
        mats.append(np.asarray(packed['lidar2img'][0], dtype=np.float32))
        distortion.append(camera['distortion'])
        frame['vis_images'].append(np.asarray(packed['img'][0], dtype=np.uint8))
    return images, intrinsics, extrinsics, mats, distortion


def render(frame, boxes, labels, scores, occ, lidar2img, bounds, class_names, z_bounds, camera_names):
    ground = other = None
    if occ is not None:
        ground, other = split_ground(occ)
    tiles = []
    for name, image, mat in zip(camera_names, frame['vis_images'], lidar2img):
        view = image.copy()
        vis_mini.draw_boxes_on_image(view, boxes, labels, mat, class_names, scores=scores)
        if ground is not None:
            vis_mini.draw_occ_on_image(view, ground, bounds, z_bounds, mat, alpha=0.55)
            vis_mini.draw_occ_on_image(view, other, bounds, z_bounds, mat)
        cv2.putText(view, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(view)
    if len(tiles) != 4:
        raise RuntimeError('渲染按四路相机排列，当前是 {}'.format(len(tiles)))
    # vis 图已经是网络输入 512×640，按这个尺寸拼，不再压到 640×360。
    views = np.concatenate([
        np.concatenate(tiles[:2], axis=1),
        np.concatenate(tiles[2:], axis=1),
    ], axis=0)
    bev_side = views.shape[0]
    bev = np.zeros((bev_side, bev_side, 3), dtype=np.uint8)
    if ground is not None:
        paint_occ_columns(bev, ground, bounds)
    if other is not None:
        paint_occ_columns(bev, other, bounds)
    paint_lidar_bev(bev, frame.get('points'), bounds, z_bounds)
    vis_mini.draw_axes(bev, bounds)
    vis_mini.draw_boxes(bev, boxes, labels, bounds, scores=scores)
    return np.concatenate([views, bev], axis=1)


def split_ground(occ):
    """地面和其余占用分开。地面铺在下面，障碍叠在上面。"""
    ground_hit = np.isin(occ, GROUND_LABELS)
    ground = np.full_like(occ, vis_mini.EMPTY)
    other = np.full_like(occ, vis_mini.EMPTY)
    ground[ground_hit] = occ[ground_hit]
    other_hit = (occ != vis_mini.EMPTY) & ~ground_hit
    other[other_hit] = occ[other_hit]
    return ground, other


def paint_occ_columns(canvas, occ, bounds):
    """每列取最高的非空体素，按和点云相同的俯视映射铺满格子。"""
    occupied = occ != vis_mini.EMPTY
    if not np.any(occupied):
        return
    top = occupied.shape[2] - 1 - np.argmax(occupied[:, :, ::-1], axis=2)
    labels = np.full(occ.shape[:2], vis_mini.EMPTY, dtype=np.int32)
    hit = occupied.any(axis=2)
    xs, ys = np.nonzero(hit)
    labels[xs, ys] = occ[xs, ys, top[xs, ys]]
    height, width = canvas.shape[:2]
    xmin, ymin, xmax, ymax = bounds
    cols = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    px, py = np.meshgrid(cols, rows)
    world_y = ymax - px / max(width - 1, 1) * (ymax - ymin)
    world_x = xmax - py / max(height - 1, 1) * (xmax - xmin)
    nx, ny = labels.shape
    ix = np.floor((world_x - xmin) / (xmax - xmin) * nx).astype(np.int32)
    iy = np.floor((world_y - ymin) / (ymax - ymin) * ny).astype(np.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    sampled = labels[np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)]
    paint = valid & (sampled != vis_mini.EMPTY)
    if not np.any(paint):
        return
    canvas[paint] = vis_mini.OCC_COLORS[sampled][..., ::-1][paint]


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
    checkpoint = args.checkpoint or cfg.load_from
    data_root = args.data_root or cfg.demo_data_root
    calib_path = args.calib or cfg.demo_calib
    camera_names = list(cfg.demo_cameras)
    cameras, lidar_poses = load_sensors(calib_path, camera_names)
    frames = index_frames(data_root, camera_names)
    stride = max(int(args.stride), 1)
    picked = list(range(0, len(frames), stride))
    if args.max_frames > 0:
        picked = picked[:args.max_frames]
    for name, camera in zip(camera_names, cameras):
        position = np.linalg.inv(camera['ego2cam'])[:3, 3]
        print('{} {} {} 在 base_link 下 {}'.format(
            name, camera['model'], camera['size'], np.round(position, 3).tolist()))
    print('两路雷达合成到 base_link，共 {} 帧，每 {} 帧渲染 1 帧，本次 {} 帧，输出 {}'.format(
        len(frames), stride, len(picked), args.out_dir))
    if not cfg.model.get('use_lidar', False):
        print('当前权重 use_lidar 为关，点云只画在 BEV 上，不进网络')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_pretrained(model, checkpoint)
    model.to(device)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    final_dim = tuple(cfg.ida_aug_conf['final_dim'])
    # 内参已经变到 final_dim 的像素系，IPM 归一化必须用同一高宽。
    if model.img_view_transformer is not None:
        model.img_view_transformer.input_size = final_dim
    print('网络输入高×宽 {}，IPM 按这个尺寸用缩放后的内参'.format(final_dim))
    occ_range = [float(v) for v in cfg.occ_range]
    bounds = (occ_range[0], occ_range[1], occ_range[3], occ_range[4])
    z_bounds = (occ_range[2], occ_range[5])
    class_names = list(cfg.class_names)
    records = []
    for index in picked:
        frame = frames[index]
        images, intrinsics, extrinsics, mats, distortion = prepare_images(
            frame, cameras, final_dim, mean, std)
        points = merge_lidars(frame, lidar_poses)
        frame['points'] = points
        data = dict(
            img=torch.from_numpy(np.stack(images)).unsqueeze(0).to(device),
            points=[torch.from_numpy(points)],
            lidar2img=torch.from_numpy(np.stack(mats)).unsqueeze(0).to(device),
            intrinsics=torch.from_numpy(np.stack(intrinsics)).unsqueeze(0).to(device),
            extrinsics=torch.from_numpy(np.stack(extrinsics)).unsqueeze(0).to(device),
            distortion=torch.from_numpy(np.stack(distortion)).unsqueeze(0),
        )
        height, width = images[0].shape[-2:]
        box_type_3d, _ = get_box_type('LiDAR')
        meta = dict(
            scene_token='mydata',
            box_type_3d=box_type_3d,
            pad_shape=[(height, width, 3)] * len(cameras),
            img_shape=[(height, width, 3)] * len(cameras),
        )
        with torch.no_grad():
            result = model.simple_test([meta], **data)[0]
        pred = result.get('pts_bbox') or {}
        boxes, labels, scores = filter_boxes(
            pred.get('boxes_3d'), pred.get('labels_3d'), pred.get('scores_3d'), args.score_thr)
        occ = result.get('occ') if SHOW_OCC else None
        image = render(
            frame, boxes, labels, scores, occ, mats,
            bounds, class_names, z_bounds, camera_names)
        name = '{}_{}.jpg'.format(index, frame['key'])
        cv2.imwrite(osp.join(args.out_dir, name), image)
        count = 0 if labels is None else int(len(labels))
        bottom = None if count == 0 else float(boxes.tensor[:, 2].mean())
        records.append(dict(frame=frame['key'], boxes=count, points=int(points.shape[0]), bottom_z=bottom))
        print('{} 点 {} 检测框 {} 底面z {}'.format(
            name, points.shape[0], count, '无' if bottom is None else round(bottom, 2)))
    mmcv.dump(records, osp.join(args.out_dir, 'frames.json'))


if __name__ == '__main__':
    main()
