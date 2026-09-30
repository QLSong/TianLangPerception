# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss
from torch import nn

from projects.mmdet3d_plugin.models.losses.focal_loss import CustomFocalLoss  # noqa: F401
from projects.mmdet3d_plugin.models.losses.grad_clip import clip_backward
from projects.mmdet3d_plugin.models.losses.lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.models.losses.semkitti_loss import geo_scal_loss, sem_scal_loss

nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027, 2074468,
    413451, 2384460, 5916653, 175883646, 4275424, 51393615, 61411620,
    105975596, 116424404, 1892500630,
])


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    """把单层 BEV 特征预测成 Dz 层占用。

    损失与 MambaOcc 的 BEVOCCHead2D_V2 相同：类别平衡 focal 再乘 100，
    semantic scale、geometric scale（空类 17）和 Lovasz-softmax。
    不使用相机可见 mask。类别权重是 1 / log(nuScenes 体素频次)。
    """
    def __init__(self, in_dim=256, out_dim=256, Dz=16, use_mask=True,
                 num_classes=18, use_predicter=True, class_balance=False,
                 loss_occ=None, perception_range=None, occ_label_range=None,
                 xy_voxel_size=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim, out_channels, kernel_size=3, stride=1, padding=1,
            bias=True, conv_cfg=dict(type='Conv2d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.ReLU(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(
                1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
        self.loss_occ = build_loss(loss_occ)
        self.occ_window = self._label_window(
            perception_range, occ_label_range, xy_voxel_size)

    @staticmethod
    def _label_window(perception_range, occ_label_range, xy_voxel_size):
        """Occ3D 的 xy 落在感知 BEV 里的下标。范围一致时不裁。"""
        if perception_range is None:
            return None
        step = float(xy_voxel_size)
        x0 = int(round((occ_label_range[0] - perception_range[0]) / step))
        y0 = int(round((occ_label_range[1] - perception_range[1]) / step))
        x1 = int(round((occ_label_range[3] - perception_range[0]) / step))
        y1 = int(round((occ_label_range[4] - perception_range[1]) / step))
        full_x = int(round((perception_range[3] - perception_range[0]) / step))
        full_y = int(round((perception_range[4] - perception_range[1]) / step))
        if min(x0, y0) < 0 or x1 > full_x or y1 > full_y or x1 <= x0 or y1 <= y0:
            raise ValueError('occ label range is outside the perception BEV')
        return full_x, full_y, x0, x1, y0, y1

    def forward(self, img_feats):
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, dx, dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, dx, dy, self.Dz, self.num_classes)
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """occ_pred 是 (B, Dx, Dy, Dz, 类)。mask_camera 保留参数以兼容调用方，不参与计算。"""
        del mask_camera
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.occ_window is not None:
            full_x, full_y, x0, x1, y0, y1 = self.occ_window
            if tuple(occ_pred.shape[1:3]) != (full_x, full_y):
                raise RuntimeError(
                    f'occ pred spatial {tuple(occ_pred.shape[1:3])} != perception BEV {(full_x, full_y)}')
            occ_pred = occ_pred[:, x0:x1, y0:y1]
            if tuple(voxel_semantics.shape[-3:]) != (x1 - x0, y1 - y0, self.Dz):
                raise RuntimeError(
                    f'occ label {tuple(voxel_semantics.shape)} != window {(x1 - x0, y1 - y0, self.Dz)}')
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()
        preds = clip_backward(preds)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0
        loss['loss_occ'] = loss_occ
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(
            preds, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(
            torch.softmax(preds, dim=1), voxel_semantics)
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        occ_res = occ_pred.softmax(-1).argmax(-1)
        return list(occ_res.cpu().numpy().astype(np.uint8))
