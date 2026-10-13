import numpy as np
import torch
import torch.nn as nn
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule

from projects.mmdet3d_plugin.models.losses.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.models.losses.semkitti_loss import geo_scal_loss, sem_scal_loss

from .cnn3d import S3DCNN
from .lift import sample_occ, scale_intrinsics, unproject_feature


@HEADS.register_module()
class GaussianOccHead(BaseModule):
    """GaussianOcc 的占用预测：图像特征抬到收缩坐标体素，3D CNN 后采样到 Occ3D。

    官方评测就是在这张体素上取出 200×200×16 的类别。3D Gaussian 光栅化依赖未编译的
    CUDA 扩展，这条路径用现有 Occ3D 标签监督体素，不走光度渲染。
    """

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 semantic_classes=18,
                 voxels_size=(24, 300, 300),
                 real_size=(-40.0, 40.0, -40.0, 40.0, -1.0, 5.4),
                 contracted_ratio=0.66667,
                 image_size=(256, 704),
                 mid_channels=32,
                 loss_occ=None):
        super().__init__()
        self.semantic_classes = semantic_classes
        self.voxels_size = tuple(int(v) for v in voxels_size)
        self.real_size = tuple(float(v) for v in real_size)
        self.contracted_ratio = float(contracted_ratio)
        self.image_size = tuple(int(v) for v in image_size)
        self.pc_range = [
            self.real_size[0], self.real_size[2], self.real_size[4],
            self.real_size[1], self.real_size[3], self.real_size[5]]
        self.proj = nn.Conv2d(in_channels, feat_channels, kernel_size=1)
        self.agg = S3DCNN(feat_channels, semantic_classes, mid_channels)
        if loss_occ is None:
            loss_occ = dict(
                type='CrossEntropyLoss', use_sigmoid=False, ignore_index=255, loss_weight=1.0)
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats, extrinsics, intrinsics, image_hw=None):
        if img_feats.dim() == 6:
            img_feats = img_feats[:, -1]
        if extrinsics.dim() == 5:
            extrinsics = extrinsics[:, -1]
            intrinsics = intrinsics[:, -1]
        batch, views, channels, height, width = img_feats.shape
        feat = self.proj(img_feats.reshape(batch * views, channels, height, width))
        feat = feat.view(batch, views, feat.shape[1], height, width)
        image_h, image_w = self.image_size if image_hw is None else image_hw
        scaled_k = scale_intrinsics(
            intrinsics.reshape(batch * views, 4, 4),
            width / float(image_w), height / float(image_h))
        ego_to_cam = extrinsics.reshape(batch * views, 4, 4)
        ego_to_pix = torch.matmul(scaled_k, ego_to_cam)
        volume = unproject_feature(
            feat, ego_to_pix, ego_to_cam, self.voxels_size, self.pc_range, self.contracted_ratio)
        logits = self.agg(volume)
        depth, y_size, x_size = self.voxels_size
        return sample_occ(
            logits, self.pc_range, self.contracted_ratio, (x_size, y_size, depth))

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        voxel_semantics = voxel_semantics.long()
        preds = occ_pred.reshape(-1, self.semantic_classes)
        target = voxel_semantics.reshape(-1)
        if mask_camera is None:
            loss_occ = self.loss_occ(preds, target)
        else:
            weight = mask_camera.reshape(-1).float()
            loss_occ = self.loss_occ(preds, target, weight=weight, avg_factor=weight.sum().clamp(min=1))
        loss_occ = loss_occ * 100.0
        return dict(
            loss_occ=loss_occ,
            loss_voxel_sem_scal=sem_scal_loss(occ_pred, voxel_semantics),
            loss_voxel_geo_scal=geo_scal_loss(occ_pred, voxel_semantics, non_empty_idx=17),
            loss_voxel_lovasz=lovasz_softmax(torch.softmax(occ_pred, dim=1), voxel_semantics))

    def get_occ(self, occ_pred, img_metas=None):
        occ = occ_pred.softmax(dim=1).argmax(dim=1)
        return list(occ.detach().cpu().numpy().astype(np.uint8))
