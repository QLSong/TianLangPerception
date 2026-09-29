import torch
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttention, MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE


def ms_deform_attn(attn, query, value, identity, query_pos, reference_points,
                   spatial_shapes, level_start_index, valid=None):
    """mmcv MultiScaleDeformableAttention 的前向，可按层屏蔽无效采样点。

    query、value 与 attn.batch_first 一致。reference_points 是
    (batch, num_query, num_levels, 2)，范围 [0, 1]，最后一维是 (x, y)。
    valid 是 (batch, num_query, num_levels)，False 的层不参与 softmax。
    """
    if value is None:
        value = query
    if identity is None:
        identity = query
    if query_pos is not None:
        query = query + query_pos
    if not attn.batch_first:
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

    bs, num_query, _ = query.shape
    _, num_value, _ = value.shape
    if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != num_value:
        raise ValueError(
            'value 长度 {} 和 spatial_shapes {} 对不上'.format(
                num_value, spatial_shapes.tolist()))

    value = attn.value_proj(value)
    value = value.view(bs, num_value, attn.num_heads, -1)
    sampling_offsets = attn.sampling_offsets(query).view(
        bs, num_query, attn.num_heads, attn.num_levels, attn.num_points, 2)
    logits = attn.attention_weights(query).view(
        bs, num_query, attn.num_heads, attn.num_levels, attn.num_points)
    if valid is not None:
        fill = -1e4 if logits.dtype == torch.float16 else -1e9
        logits = logits.masked_fill(~valid[:, :, None, :, None], fill)
    attention_weights = logits.reshape(
        bs, num_query, attn.num_heads, attn.num_levels * attn.num_points).softmax(-1)
    attention_weights = attention_weights.view(
        bs, num_query, attn.num_heads, attn.num_levels, attn.num_points)
    if valid is not None:
        all_invalid = ~valid.any(dim=-1)
        attention_weights = attention_weights.masked_fill(
            all_invalid[:, :, None, None, None], 0.0)

    reference_points = reference_points.to(dtype=sampling_offsets.dtype)
    if valid is not None:
        reference_points = reference_points.masked_fill(~valid[..., None], 0.0)
    offset_normalizer = torch.stack(
        [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    sampling_locations = (
        reference_points[:, :, None, :, None, :]
        + sampling_offsets / offset_normalizer[None, None, None, :, None, :])

    if ((IS_CUDA_AVAILABLE and value.is_cuda)
            or (IS_MLU_AVAILABLE and value.is_mlu)):
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, attn.im2col_step)
    else:
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)

    output = attn.output_proj(output)
    if not attn.batch_first:
        output = output.permute(1, 0, 2)
    return attn.dropout(output) + identity


class BEVDeformCrossAttention(BaseModule):
    """对单张 BEV 特征做可变形交叉注意力。reference 的 xy 直接对应 BEV 网格。"""

    def __init__(self, embed_dims=256, num_heads=8, num_points=4, dropout=0.1,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=1,
            num_points=num_points, dropout=dropout, batch_first=False)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                reference_points=None, spatial_hw=None, **kwargs):
        del key, kwargs
        h, w = spatial_hw
        ref = reference_points[:, :, None, :]
        spatial_shapes = torch.tensor(
            [[h, w]], device=query.device, dtype=torch.long)
        level_start = torch.zeros(1, device=query.device, dtype=torch.long)
        return ms_deform_attn(
            self.attn, query, value, identity, query_pos, ref,
            spatial_shapes, level_start)


@ATTENTION.register_module()
class ImageDeformCrossAttention(BaseModule):
    """把 3D reference 投到每个相机特征图上，再做可变形交叉注意力。

    每个相机是一个 level。投到图像外或相机后方的 level 不参与 softmax。
    """

    def __init__(self, embed_dims=256, num_heads=8, num_points=4, num_cams=6,
                 dropout=0.1, pc_range=None, init_cfg=None, **kwargs):
        super().__init__(init_cfg)
        del kwargs
        if pc_range is None:
            raise ValueError('ImageDeformCrossAttention 需要 pc_range')
        self.num_cams = num_cams
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32),
            persistent=False)
        self.attn = MultiScaleDeformableAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=num_cams,
            num_points=num_points, dropout=dropout, batch_first=False)

    def _project(self, reference_points, lidar2img, img_pad):
        # reference_points: (B, Q, 3)，按 pc_range 归一化。lidar2img: (B, N, 4, 4)。
        span = self.pc_range[3:6] - self.pc_range[0:3]
        xyz = reference_points * span + self.pc_range[0:3]
        hom = torch.cat([xyz, xyz.new_ones(xyz.shape[0], xyz.shape[1], 1)], dim=-1)
        cam = torch.matmul(
            lidar2img[:, :, None, :, :], hom[:, None, :, :, None]).squeeze(-1)
        depth = cam[..., 2]
        uv = cam[..., :2] / depth.clamp(min=1e-5).unsqueeze(-1)
        pad_h, pad_w = img_pad
        u = uv[..., 0] / pad_w
        v = uv[..., 1] / pad_h
        ref = torch.stack((u, v), dim=-1).permute(0, 2, 1, 3).contiguous()
        valid = (depth > 1e-5) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
        return ref, valid.permute(0, 2, 1).contiguous()

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                reference_points=None, lidar2img=None, img_hw=None, img_pad=None,
                **kwargs):
        del key, kwargs
        if reference_points is None or lidar2img is None or img_pad is None:
            raise ValueError(
                'ImageDeformCrossAttention 需要 reference_points、lidar2img、img_pad')
        n, h, w = img_hw
        if lidar2img.shape[1] != n or n != self.num_cams:
            raise ValueError(
                '相机数 {} 和注意力 level 数 {} 不一致'.format(n, self.num_cams))
        ref, valid = self._project(reference_points, lidar2img, img_pad)
        spatial_shapes = torch.tensor(
            [[h, w]] * n, device=query.device, dtype=torch.long)
        level_start = (
            torch.arange(n, device=query.device, dtype=torch.long) * (h * w))
        return ms_deform_attn(
            self.attn, query, value, identity, query_pos, ref,
            spatial_shapes, level_start, valid)
