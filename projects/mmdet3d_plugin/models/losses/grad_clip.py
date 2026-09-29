"""损失反传的梯度截断。前向数值不变。"""
import torch


class _ClipBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, limit):
        ctx.limit = float(limit)
        return tensor

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        limit = ctx.limit
        finite = torch.isfinite(grad)
        clipped = grad.clamp(-limit, limit)
        return torch.where(finite, clipped, torch.zeros_like(grad)), None


def clip_backward(tensor, limit=100.0):
    """前向原样返回。反传时非有限值清成 0，单个元素限制在 ±limit。

    不按张量范数缩放。参数梯度的范数仍由优化器的 grad_clip 截断。
    """
    return _ClipBackward.apply(tensor, float(limit))
