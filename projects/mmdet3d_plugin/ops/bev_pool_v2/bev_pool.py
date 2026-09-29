# Copyright (c) Phigent Robotics. All rights reserved.
"""BEVPoolv2.

Training uses a differentiable index_add so no CUDA extension is required.
ONNX export keeps the official mmdeploy::bev_pool_v2 symbolic used on Orin.
"""
import torch

__all__ = ['bev_pool_v2', 'TRTBEVPoolv2']


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    """Scatter depth-weighted features into the BEV grid.

    Args:
        depth: (B, N, D, fH, fW)
        feat:  (B, N, fH, fW, C)
        ranks_*: flat indices produced by LSSViewTransformer
        bev_feat_shape: (B, Dz, Dy, Dx, C)
    Returns:
        (B, C, Dz, Dy, Dx)
    """
    del interval_starts, interval_lengths
    b, dz, dy, dx, c = bev_feat_shape
    depth_flat = depth.reshape(-1)
    feat_flat = feat.reshape(-1, feat.shape[-1])
    d = depth_flat.index_select(0, ranks_depth.long())
    f = feat_flat.index_select(0, ranks_feat.long())
    val = f * d.unsqueeze(-1)
    out = feat.new_zeros(b * dz * dy * dx, c)
    out = out.index_add(0, ranks_bev.long(), val)
    out = out.view(b, dz, dy, dx, c)
    return out.permute(0, 4, 1, 2, 3).contiguous()


class TRTBEVPoolv2(torch.autograd.Function):
    """Official symbolic. Orin engine replaces this with the bev_pool_v2 plugin."""

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                 interval_starts, interval_lengths,
                 output_height=128, output_width=128, output_z=1):
        return g.op(
            'mmdeploy::bev_pool_v2',
            depth, feat, ranks_depth, ranks_feat, ranks_bev,
            interval_starts, interval_lengths,
            output_height_i=output_height,
            output_width_i=output_width,
            output_z_i=output_z)

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                interval_starts, interval_lengths,
                output_height=128, output_width=128, output_z=1):
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():
            c = feat.shape[-1]
            if output_z == 1:
                return feat.new_zeros(1, output_height, output_width, c)
            return feat.new_zeros(1, c, output_z, output_height, output_width)
        feat = feat.unsqueeze(0)
        depth = depth.unsqueeze(0)
        bev_feat_shape = (depth.shape[0], output_z, output_height,
                          output_width, feat.shape[-1])
        bev_feat = bev_pool_v2(
            depth, feat, ranks_depth, ranks_feat, ranks_bev,
            bev_feat_shape, interval_starts, interval_lengths)
        if output_z == 1:
            bev_feat = bev_feat.squeeze(2).permute(0, 2, 3, 1)
        return bev_feat
