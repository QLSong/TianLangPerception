import os
import os.path as osp
import pickle
from concurrent.futures import ThreadPoolExecutor

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type
from nuscenes.eval.common.utils import Quaternion

from projects.mmdet3d_plugin.datasets.nuscenes_dataset import transform_points_lidar_to_ego

_OCC_INDEX = {}


def _index_occ_root(occ_root):
    """Map sample token -> labels.npz. Occ3D layout is gts/scene-xxxx/token/labels.npz."""
    cached = _OCC_INDEX.get(occ_root)
    if cached is not None:
        return cached
    index = {}
    for dirpath, _, filenames in os.walk(occ_root):
        if 'labels.npz' in filenames:
            index[osp.basename(dirpath)] = osp.join(dirpath, 'labels.npz')
    _OCC_INDEX[occ_root] = index
    return index


@PIPELINES.register_module()
class LoadOcc3DFromFile:
    """Load Occ3D-nuScenes semantics and camera visibility mask."""

    def __init__(self, occ_root):
        self.occ_root = occ_root

    def __call__(self, results):
        token = results.get('sample_idx', results.get('sample_token'))
        scene_name = results.get('scene_name')
        if scene_name:
            path = osp.join(self.occ_root, scene_name, token, 'labels.npz')
        else:
            path = _index_occ_root(self.occ_root).get(token)
        if path is None or not osp.isfile(path):
            raise FileNotFoundError(
                f'Occ3D label not found for sample {token} under {self.occ_root}')
        occ = np.load(path)
        results['voxel_semantics'] = np.ascontiguousarray(occ['semantics'])
        results['mask_camera'] = np.ascontiguousarray(occ['mask_camera'].astype(bool))
        return results


def _read_image(path, color_type):
    return mmcv.imread(path, color_type)


def _read_points(path, load_dim, use_dim):
    points = np.fromfile(path, dtype=np.float32)
    if points.size == 0 or points.size % load_dim != 0:
        raise ValueError('点云 {} 的长度 {} 不能按 {} 维切开'.format(
            path, points.size, load_dim))
    return points.reshape(-1, load_dim)[:, use_dim]


@PIPELINES.register_module()
class LoadMultiViewImageAndPoints:
    """6 路图像互相并行读，并和激光雷达同时读。"""

    def __init__(self, to_float32=False, color_type='unchanged',
                 load_points=True, coord_type='LIDAR', load_dim=5, use_dim=4):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.load_points = load_points
        self.coord_type = coord_type
        self.load_dim = load_dim
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self._executor = None

    def _pool(self, n_jobs):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max(n_jobs, 1))
        return self._executor

    def __call__(self, results):
        filename = results['img_filename']
        pool = self._pool(len(filename) + (1 if self.load_points else 0))
        image_jobs = [pool.submit(_read_image, name, self.color_type) for name in filename]
        point_job = None
        if self.load_points:
            point_job = pool.submit(
                _read_points, results['pts_filename'], self.load_dim, self.use_dim)
        images = [job.result() for job in image_jobs]
        img = np.stack(images, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        if point_job is not None:
            points = point_job.result()
            lidar2ego = results.get('lidar2ego')
            if lidar2ego is not None:
                points = transform_points_lidar_to_ego(points, lidar2ego)
            points_class = get_points_type(self.coord_type)
            results['points'] = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
        return results


# 伪标签通道到 Occ3D 类别。和 GaussianOcc nusc_dataset.semantic_map 一致。
_SEMANTIC_MAP = np.array([
    0, 4, 11, 3, 10, 14, 16, 13, 2, 1, 7, 15, 6, 5, 9, 8, 17,
], dtype=np.int64)
_CAM_CHAIN = {}


def _pose_matrix(translation, rotation):
    rot = Quaternion(np.asarray(rotation)).rotation_matrix.astype(np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(translation, dtype=np.float32)
    return mat


def _chain_cache_candidates(data_root):
    return [
        osp.join('work_dirs', 'gaussianocc_cam_chain.pkl'),
        osp.join(data_root, 'gaussianocc_cam_chain.pkl'),
    ]


def _load_chain_cache(data_root):
    for path in _chain_cache_candidates(data_root):
        if osp.isfile(path):
            with open(path, 'rb') as handle:
                return pickle.load(handle)
    return None


def _save_chain_cache(chain, data_root):
    for path in _chain_cache_candidates(data_root):
        try:
            os.makedirs(osp.dirname(path) or '.', exist_ok=True)
            with open(path, 'wb') as handle:
                pickle.dump(chain, handle, protocol=4)
            return
        except OSError:
            continue


def _cam_chain(data_root):
    """sample_data token -> 文件名、前后 token、ego2global、cam2ego。第一次会扫 nuScenes 表。"""
    cached = _CAM_CHAIN.get(data_root)
    if cached is not None:
        return cached
    chain = _load_chain_cache(data_root)
    if chain is not None:
        _CAM_CHAIN[data_root] = chain
        return chain
    import json
    table = osp.join(data_root, 'v1.0-trainval')
    with open(osp.join(table, 'ego_pose.json'), 'r') as handle:
        ego_rows = json.load(handle)
    ego = {row['token']: (row['translation'], row['rotation']) for row in ego_rows}
    del ego_rows
    with open(osp.join(table, 'calibrated_sensor.json'), 'r') as handle:
        calib_rows = json.load(handle)
    calib = {row['token']: (row['translation'], row['rotation']) for row in calib_rows}
    del calib_rows
    with open(osp.join(table, 'sample_data.json'), 'r') as handle:
        sample_rows = json.load(handle)
    chain = {}
    for row in sample_rows:
        if 'CAM' not in row['filename']:
            continue
        chain[row['token']] = (
            row['filename'],
            row['prev'] or '',
            row['next'] or '',
            _pose_matrix(*ego[row['ego_pose_token']]),
            _pose_matrix(*calib[row['calibrated_sensor_token']]))
    del sample_rows, ego, calib
    _save_chain_cache(chain, data_root)
    _CAM_CHAIN[data_root] = chain
    return chain


def _semantic_file(data_root, img_path):
    name = osp.basename(img_path)
    cam = osp.basename(osp.dirname(img_path))
    split = osp.basename(osp.dirname(osp.dirname(img_path)))
    return osp.join(data_root, 'nuscenes_semantic', split, cam, name[:-4] + '_mask.bin')


def _load_semantic(path):
    label = np.fromfile(path, dtype=np.int8)
    if label.size != 900 * 1600:
        raise RuntimeError('语义标签 {} 的长度是 {}，不是 900×1600'.format(path, label.size))
    return _SEMANTIC_MAP[label.astype(np.int64)].reshape(900, 1600)


def _as_uint8(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _cam0_to_cami(cam0_to_ego, ego0_to_global, neighbor):
    ego_i = neighbor[3]
    cam_i = neighbor[4]
    ego_i_to_cam = np.linalg.inv(cam_i)
    global_to_ego_i = np.linalg.inv(ego_i)
    return (ego_i_to_cam @ global_to_ego_i @ ego0_to_global @ cam0_to_ego).astype(np.float32)


@PIPELINES.register_module()
class LoadGaussianOccSupervise:
    """读 nusc-sem-gs 的监督：当前帧二维语义，以及每个相机 12Hz 的前一帧和后一帧。"""

    def __init__(self, data_root):
        self.data_root = data_root

    def __call__(self, results):
        chain = _cam_chain(self.data_root)
        images_m1 = []
        images_p1 = []
        semantics = []
        pose_m1 = []
        pose_p1 = []
        for index, token in enumerate(results['cam_sd_token']):
            img_path = results['img_filename'][index]
            sem_path = _semantic_file(self.data_root, img_path)
            if not osp.isfile(sem_path):
                raise FileNotFoundError('没有找到二维语义 {}'.format(sem_path))
            semantics.append(_load_semantic(sem_path))
            cur = _as_uint8(results['img'][index])
            cur_ego = np.asarray(results['cam_ego2global'][index], dtype=np.float32)
            cur_cam = np.asarray(results['cam_to_ego'][index], dtype=np.float32)
            record = chain.get(token)
            if record is None:
                raise KeyError('相机 sample_data {} 不在索引里'.format(token))
            for slot, images, poses in ((1, images_m1, pose_m1), (2, images_p1, pose_p1)):
                other = record[slot]
                neighbor = chain.get(other) if other else None
                if neighbor is None:
                    images.append(cur)
                    poses.append(np.eye(4, dtype=np.float32))
                    continue
                path = osp.join(self.data_root, neighbor[0])
                if not osp.isfile(path):
                    raise FileNotFoundError('没有找到相邻帧 {}'.format(path))
                images.append(mmcv.imread(path, 'unchanged'))
                poses.append(_cam0_to_cami(cur_cam, cur_ego, neighbor))
        results['gs_m1_img'] = images_m1
        results['gs_p1_img'] = images_p1
        results['gs_semantic_full'] = semantics
        results['gs_cam_T_m1'] = np.stack(pose_m1, axis=0)
        results['gs_cam_T_p1'] = np.stack(pose_p1, axis=0)
        return results
