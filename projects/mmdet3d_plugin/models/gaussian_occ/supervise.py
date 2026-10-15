"""nusc-sem-gs 的自监督：光度重投影、边缘平滑、二维语义交叉熵。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# options.py 的默认 class_frequencies。nusc-sem-gs 没有改这一项。
# 频率为 0 的类权重变成 0，二维伪标签里的 ignore 就是这样丢掉的。
CLASS_FREQUENCIES = [
    0, 18164955800, 734218842, 2336187448, 10996756106, 395414611,
    638889260, 651023279, 117046208, 985341947, 7303776233, 43131984997,
    0, 9342674867, 9525824742, 51204832885, 22848065525, 31841090018,
]


def class_weights(num_classes):
    freq = np.array(CLASS_FREQUENCIES[:num_classes], dtype=np.float32)
    if freq.shape[0] != num_classes:
        raise ValueError('class_frequencies 长度 {} 和类别数 {} 不一致'.format(
            freq.shape[0], num_classes))
    with np.errstate(divide='ignore'):
        weight = 1.0 / np.sqrt(freq)
    weight = np.nan_to_num(weight, posinf=0.0)
    weight = weight / np.mean(weight)
    return weight


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        numer = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        denom = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - numer / denom) / 2, 0, 1)


def smooth_loss(disp, image):
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_x = grad_disp_x * torch.exp(-grad_img_x)
    grad_disp_y = grad_disp_y * torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def _backproject(depth, inv_k):
    n_view, _, height, width = depth.shape
    xs = torch.arange(width, device=depth.device, dtype=depth.dtype)
    ys = torch.arange(height, device=depth.device, dtype=depth.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    pix = torch.stack([
        grid_x.reshape(-1), grid_y.reshape(-1),
        torch.ones(height * width, device=depth.device, dtype=depth.dtype),
    ], dim=0).unsqueeze(0)
    cam = torch.matmul(inv_k[:, :3, :3], pix)
    cam = depth.reshape(n_view, 1, -1) * cam
    ones = torch.ones(n_view, 1, height * width, device=depth.device, dtype=depth.dtype)
    return torch.cat([cam, ones], dim=1)


def _project(points, intrinsics, transform, height, width):
    camera = torch.matmul(torch.matmul(intrinsics, transform)[:, :3, :], points)
    pix = camera[:, :2, :] / (camera[:, 2:3, :] + 1e-7)
    pix = pix.view(-1, 2, height, width).permute(0, 2, 3, 1)
    pix = pix.clone()
    pix[..., 0] = pix[..., 0] / (width - 1)
    pix[..., 1] = pix[..., 1] / (height - 1)
    return (pix - 0.5) * 2


def reprojection_loss(pred, target, ssim):
    l1_loss = torch.abs(target - pred).mean(1, keepdim=True)
    return 0.85 * ssim(pred, target).mean(1, keepdim=True) + 0.15 * l1_loss


def photometric_loss(depth, color, sources, transforms, intrinsics, ssim, smoothness):
    """depth、color 都在网络输入分辨率。sources / transforms 是前后两帧。"""
    n_view, _, height, width = depth.shape
    inv_k = torch.linalg.inv(intrinsics)
    points = _backproject(depth, inv_k)
    warped = []
    identity = []
    for source, transform in zip(sources, transforms):
        pix = _project(points, intrinsics, transform, height, width)
        pred = F.grid_sample(source, pix, padding_mode='border', align_corners=True)
        warped.append(reprojection_loss(pred, color, ssim))
        identity.append(reprojection_loss(source, color, ssim))
    warped = torch.cat(warped, dim=1)
    identity = torch.cat(identity, dim=1)
    identity = identity + torch.randn_like(identity) * 1e-5
    combined = torch.cat([identity, warped], dim=1)
    photo = combined.min(dim=1).values.mean()
    disp = 1.0 / (depth + 1e-7)
    disp = disp / (disp.mean(dim=(2, 3), keepdim=True) + 1e-7)
    return photo + smoothness * smooth_loss(disp, color)


def semantic_ce(pred, target, weight, loss_weight):
    """pred 是 (N, H, W, 类)，target 是 (N, H, W) 的类别 id。"""
    n_view, height, width, classes = pred.shape
    label = F.interpolate(
        target.float().view(n_view, 1, target.shape[-2], target.shape[-1]),
        size=(height, width), mode='nearest').view(-1).long()
    logits = pred.reshape(-1, classes)
    loss = F.cross_entropy(logits, label, weight=weight, ignore_index=-1)
    return loss_weight * loss


def opacity_entropy(logits, loss_weight):
    """3D Gaussian 的不透明度熵。last channel 的占据概率是 1 - sigmoid。"""
    opacity = 1.0 - torch.sigmoid(logits[:, -1])
    loss = -(opacity * torch.log(opacity + 1e-8)).mean()
    return loss_weight * loss


def voxel_tv(logits, loss_weight):
    """nusc-sem-gs 的 distortion：对体素特征做 total variation。"""
    tv_x = logits.diff(dim=2).abs().mean()
    tv_y = logits.diff(dim=3).abs().mean()
    tv_z = logits.diff(dim=4).abs().mean()
    return loss_weight * (tv_x + tv_y + tv_z) / 3.0
