import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule

from projects.mmdet3d_plugin.models.losses.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.models.losses.semkitti_loss import geo_scal_loss, sem_scal_loss

from .cnn3d import S3DCNN
from .lift import sample_occ, scale_intrinsics, unproject_feature
from .render import ray_samples, render_views
from .supervise import SSIM, class_weights, opacity_entropy, photometric_loss, semantic_ce, voxel_tv


@HEADS.register_module()
class GaussianOccHead(BaseModule):
    """GaussianOcc 的占用预测：图像特征抬到收缩坐标体素，3D CNN 后采样到 Occ3D。

    supervise='self' 对齐 nusc-sem-gs：光度重投影、二维语义、不透明度熵、体素 TV。
    评测仍把体素采样成 200×200×16。3D Gaussian 光栅化扩展没装，训练时的深度和语义图
    用官方 prob 射线积分。
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
                 supervise='self',
                 semantic_loss_weight=0.02,
                 disparity_smoothness=0.001,
                 weight_entropy_last=0.1,
                 weight_distortion=0.1,
                 semantic_sample_ratio=0.25,
                 last_free=True,
                 min_depth=0.1,
                 max_depth=80.0,
                 stepsize=0.5,
                 render_size=(128, 352),
                 loss_occ=None):
        super().__init__()
        self.semantic_classes = semantic_classes
        self.voxels_size = tuple(int(v) for v in voxels_size)
        self.real_size = tuple(float(v) for v in real_size)
        self.contracted_ratio = float(contracted_ratio)
        self.image_size = tuple(int(v) for v in image_size)
        self.supervise = supervise
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.disparity_smoothness = float(disparity_smoothness)
        self.weight_entropy_last = float(weight_entropy_last)
        self.weight_distortion = float(weight_distortion)
        self.semantic_sample_ratio = float(semantic_sample_ratio)
        self.last_free = bool(last_free)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.stepsize = float(stepsize)
        self.render_size = (int(render_size[0]), int(render_size[1]))
        self.pc_range = [
            self.real_size[0], self.real_size[2], self.real_size[4],
            self.real_size[1], self.real_size[3], self.real_size[5]]
        self.proj = nn.Conv2d(in_channels, feat_channels, kernel_size=1)
        self.agg = S3DCNN(feat_channels, semantic_classes, mid_channels)
        self.ssim = SSIM()
        z_base, voxel_size = ray_samples(
            self.real_size, self.voxels_size, self.contracted_ratio, self.stepsize)
        self.register_buffer('z_base', z_base, persistent=False)
        self.ray_voxel_size = voxel_size
        self.register_buffer(
            'sem_weight', torch.tensor(class_weights(semantic_classes)), persistent=False)
        if loss_occ is None:
            loss_occ = dict(
                type='CrossEntropyLoss', use_sigmoid=False, ignore_index=255, loss_weight=1.0)
        self.loss_occ = build_loss(loss_occ)

    def _logits(self, img_feats, extrinsics, intrinsics, image_hw=None):
        if img_feats.dim() == 6:
            img_feats = img_feats[:, -1]
        if extrinsics.dim() == 5:
            extrinsics = extrinsics[:, -1]
            intrinsics = intrinsics[:, -1]
        preds = [
            self._logits_one(
                img_feats[index:index + 1],
                extrinsics[index:index + 1],
                intrinsics[index:index + 1],
                image_hw)
            for index in range(img_feats.shape[0])
        ]
        return torch.cat(preds, dim=0), extrinsics, intrinsics

    def _logits_one(self, img_feats, extrinsics, intrinsics, image_hw):
        """一个样本走完抬升和 3D CNN。训练时丢掉中间激活，反传再重算这一份。"""
        if not (self.training and torch.is_grad_enabled()):
            return self._volume_logits(img_feats, extrinsics, intrinsics, image_hw)

        def run(img_feats, extrinsics, intrinsics):
            return self._volume_logits(img_feats, extrinsics, intrinsics, image_hw)

        return checkpoint(run, img_feats, extrinsics, intrinsics, use_reentrant=False)

    def _volume_logits(self, img_feats, extrinsics, intrinsics, image_hw):
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
        # 检测器的 forward 被 force_fp32 关掉了 autocast，这里把 3D 卷积拉回半精度。
        with torch.cuda.amp.autocast(enabled=volume.is_cuda):
            return self.agg(volume)

    def forward(self, img_feats, extrinsics, intrinsics, image_hw=None):
        logits, _, _ = self._logits(img_feats, extrinsics, intrinsics, image_hw)
        depth, y_size, x_size = self.voxels_size
        return sample_occ(
            logits, self.pc_range, self.contracted_ratio, (x_size, y_size, depth))

    def loss_self(self, img_feats, extrinsics, intrinsics, color, color_m1, color_p1,
                  cam_t_m1, cam_t_p1, semantic):
        """nusc-sem-gs 的 self 监督。各损失已经乘过配置里的权重。"""
        extrinsics = self._drop_time(extrinsics, 4)
        intrinsics = self._drop_time(intrinsics, 4)
        color = self._drop_time(color, 5)
        color_m1 = self._drop_time(color_m1, 5)
        color_p1 = self._drop_time(color_p1, 5)
        cam_t_m1 = self._drop_time(cam_t_m1, 4)
        cam_t_p1 = self._drop_time(cam_t_p1, 4)
        semantic = self._drop_time(semantic, 4)
        image_h, image_w = int(color.shape[-2]), int(color.shape[-1])
        logits, extrinsics, intrinsics = self._logits(
            img_feats, extrinsics, intrinsics, (image_h, image_w))
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            extrinsics = extrinsics.float()
            intrinsics = intrinsics.float()
            color = color.float()
            color_m1 = color_m1.float()
            color_p1 = color_p1.float()
            cam_t_m1 = cam_t_m1.float()
            cam_t_p1 = cam_t_p1.float()
            semantic = semantic.long()
            render_h, render_w = self.render_size
            photo = color.new_zeros(())
            sem = color.new_zeros(())
            batch = logits.shape[0]
            for index in range(batch):
                k_image = intrinsics[index]
                k_render = k_image.clone()
                k_render[:, 0, :] *= render_w / float(image_w)
                k_render[:, 1, :] *= render_h / float(image_h)
                cam_to_ego = torch.linalg.inv(extrinsics[index])
                depth, pred_sem = render_views(
                    logits[index:index + 1], k_render, cam_to_ego, self.pc_range,
                    self.contracted_ratio, self.z_base, self.ray_voxel_size,
                    self.stepsize, self.min_depth, self.semantic_sample_ratio,
                    self.last_free, (render_h, render_w), training=self.training)
                depth = depth.clamp(min=self.min_depth, max=200.0)
                depth = F.interpolate(
                    depth, size=(image_h, image_w), mode='bilinear', align_corners=False)
                photo = photo + photometric_loss(
                    depth, color[index], [color_m1[index], color_p1[index]],
                    [cam_t_m1[index], cam_t_p1[index]], k_image, self.ssim,
                    self.disparity_smoothness)
                sem = sem + semantic_ce(
                    pred_sem, semantic[index], self.sem_weight, self.semantic_loss_weight)
            scale = float(max(batch, 1))
            return dict(
                loss_pe=photo / scale,
                loss_semantic=sem / scale,
                loss_entropy_last=opacity_entropy(logits, self.weight_entropy_last),
                loss_distortion=voxel_tv(logits, self.weight_distortion))

    @staticmethod
    def _drop_time(tensor, ndim):
        if tensor.dim() == ndim + 1:
            return tensor[:, -1]
        return tensor

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
