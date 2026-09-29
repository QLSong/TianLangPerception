import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

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
