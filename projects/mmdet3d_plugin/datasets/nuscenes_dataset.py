# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
import math
@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, seq_mode=False, seq_split_num=1, num_frame_losses=1, queue_length=8, random_length=0, *args, **kwargs):
        gsocc_supervise = kwargs.pop('gsocc_supervise', False)
        self.class_merge = kwargs.pop('class_merge', None)
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.collect_keys = collect_keys
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        if seq_mode:
            self.num_frame_losses = 1
            self.queue_length = 1
            self.seq_split_num = seq_split_num
            self.random_length = 0
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        if gsocc_supervise:
            self._attach_gsocc_supervise()

    def _attach_gsocc_supervise(self):
        """只在 GaussianOcc 训练集上加前后帧和二维语义。pipeline 在配置合并后才插得进去。"""
        from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadGaussianOccSupervise
        transforms = self.pipeline.transforms
        loader = LoadGaussianOccSupervise(data_root=self.data_root)
        insert_at = next(
            i for i, step in enumerate(transforms)
            if step.__class__.__name__ == 'ResizeCropFlipRotImage')
        transforms.insert(insert_at, loader)
        extra = [
            'gs_color', 'gs_color_m1', 'gs_color_p1', 'gs_semantic',
            'gs_cam_T_m1', 'gs_cam_T_p1']
        for step in transforms:
            if step.__class__.__name__ != 'Collect3D' or not hasattr(step, 'keys'):
                continue
            keys = list(step.keys)
            if 'gt_bboxes_3d' not in keys:
                continue
            for name in extra:
                if name not in keys:
                    keys.append(name)
            step.keys = keys

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)


    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length-self.random_length+1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            
            if not self.seq_mode: # for sliding window only
                if input_dict['scene_token'] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict['scene_token']
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            queue.append(example)

        for k in range(self.num_frame_losses):
            if self.filter_empty_gt and \
                (queue[-k-1] is None or ~(queue[-k-1]['gt_labels_3d']._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        # 点、框和相机外参都变到自车坐标后再做时序，位姿是 ego2global。
        ego_pose = e2g_matrix
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        ego2lidar = invert_matrix_egopose_numpy(l2e_matrix)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego=l2e_matrix,
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)
                # 自车 -> 相机。LIDAR_TOP 相对车体有 -90° 偏航，不能直接拿激光雷达轴当 x 朝前。
                ego2cam_rt = lidar2cam_rt @ ego2lidar

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ ego2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(ego2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
            cam_ego2global = []
            cam_to_ego = []
            cam_sd_token = []
            for cam_info in info['cams'].values():
                cam_ego2global.append(_qt_matrix(
                    cam_info['ego2global_rotation'], cam_info['ego2global_translation']))
                cam_to_ego.append(_qt_matrix(
                    cam_info['sensor2ego_rotation'], cam_info['sensor2ego_translation']))
                cam_sd_token.append(cam_info['sample_data_token'])
                
            if not self.test_mode: # for seq_mode
                prev_exists  = not (index == 0 or self.flag[index - 1] != self.flag[index])
            else:
                prev_exists = None

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    prev_exists=prev_exists,
                    cam_ego2global=cam_ego2global,
                    cam_to_ego=cam_to_ego,
                    cam_sd_token=cam_sd_token,
                ))
        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def get_ann_info(self, index):
        ann = super().get_ann_info(index)
        if self.class_merge:
            names = [self.class_merge.get(str(name), str(name)) for name in ann['gt_names']]
            ann['gt_names'] = np.array(names)
            ann['gt_labels_3d'] = np.array([
                self.CLASSES.index(name) if name in self.CLASSES else -1
                for name in names
            ], dtype=np.int64)
        ann['gt_bboxes_3d'] = transform_boxes_lidar_to_ego(
            ann['gt_bboxes_3d'], self.data_infos[index])
        return ann

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None,
                 result_names=['pts_bbox'], show=False, out_dir=None, pipeline=None):
        """合并过的类别不在 nuScenes 官方 10 类里，改用自车系上的检测指标。"""
        if not self.class_merge:
            return super().evaluate(
                results, metric=metric, logger=logger, jsonfile_prefix=jsonfile_prefix,
                result_names=result_names, show=show, out_dir=out_dir, pipeline=pipeline)
        from projects.mmdet3d_plugin.core.evaluation.det_metrics import evaluate_det, pack_boxes
        records = []
        for index, result in enumerate(results):
            pred = result.get('pts_bbox', result)
            ann = self.get_ann_info(index)
            records.append(dict(
                token=self.data_infos[index]['token'],
                gt=pack_boxes(ann['gt_bboxes_3d'], ann['gt_labels_3d']),
                pred=pack_boxes(pred['boxes_3d'], pred['labels_3d'], pred['scores_3d']),
            ))
        metrics, text = evaluate_det(records, list(self.CLASSES))
        print('\n' + text)
        return {
            'pts_bbox_NuScenes/mAP': float(metrics['mAP']),
            'pts_bbox_NuScenes/NDS': float(metrics['NDS']),
        }

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def format_results(self, results, jsonfile_prefix=None):
        """检测和占用写在同一个结果 dict 里。父类会遍历每个键，占用数组会被当成检测框索引。"""
        if results and isinstance(results[0], dict):
            det_keys = [key for key in ('pts_bbox', 'img_bbox') if key in results[0]]
            if det_keys:
                results = [
                    {key: sample[key] for key in det_keys if key in sample}
                    for sample in results]
        return super().format_results(results, jsonfile_prefix)

    def _format_bbox(self, results, jsonfile_prefix=None):
        """预测框已经在自车坐标。父类导出时会再乘 lidar2ego，这里改成只乘 ego2global。"""
        import mmdet3d.datasets.nuscenes_dataset as nusc_ds
        saved = nusc_ds.lidar_nusc_box_to_global
        nusc_ds.lidar_nusc_box_to_global = _ego_box_to_global
        try:
            return super()._format_bbox(results, jsonfile_prefix)
        finally:
            nusc_ds.lidar_nusc_box_to_global = saved

def _ego_box_to_global(info, boxes, classes, eval_configs, eval_version='detection_cvpr_2019'):
    """自车坐标的框滤掉感知范围外的，再变到全局。不再乘 lidar2ego。"""
    del eval_version
    box_list = []
    for box in boxes:
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = eval_configs.class_range[classes[box.label]]
        if radius > det_range:
            continue
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


def lidar_to_ego_matrix(info):
    """LIDAR_TOP 到自车。nuScenes 上这是绕 z 轴约 -90°，激光雷达 +y 才是车头。"""
    rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    translation = info['lidar2ego_translation']
    return convert_egopose_to_matrix_numpy(rotation, translation)


def transform_points_lidar_to_ego(points, lidar2ego):
    """点云 xyz 变到自车坐标，x 朝前、y 朝左。其余通道保持原样。"""
    out = np.array(points, dtype=np.float32, copy=True)
    rotation = np.asarray(lidar2ego[:3, :3], dtype=np.float32)
    translation = np.asarray(lidar2ego[:3, 3], dtype=np.float32)
    out[:, :3] = out[:, :3] @ rotation.T + translation
    return out


def transform_boxes_lidar_to_ego(boxes, info):
    """3D 框中心、朝向和速度变到自车坐标。框自身的长宽高不变。"""
    if boxes is None or len(boxes) == 0:
        return boxes
    lidar2ego = lidar_to_ego_matrix(info)
    data = boxes.tensor.detach().cpu().numpy().astype(np.float64, copy=True)
    rotation = np.asarray(lidar2ego[:3, :3], dtype=np.float64)
    translation = np.asarray(lidar2ego[:3, 3], dtype=np.float64)
    data[:, :3] = data[:, :3] @ rotation.T + translation
    yaw = data[:, 6]
    heading = np.stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)), axis=-1)
    heading = heading @ rotation.T
    data[:, 6] = np.arctan2(heading[:, 1], heading[:, 0])
    if data.shape[1] > 8:
        velocity = np.stack((data[:, 7], data[:, 8], np.zeros(len(data))), axis=-1)
        data[:, 7:9] = (velocity @ rotation.T)[:, :2]
    return boxes.new_box(data.astype(np.float32))


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def _qt_matrix(rotation, translation):
    rot = np.asarray(rotation)
    if rot.shape == (4,):
        rot = Quaternion(rot).rotation_matrix
    return convert_egopose_to_matrix_numpy(rot, translation)


def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix