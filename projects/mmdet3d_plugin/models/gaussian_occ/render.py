"""GaussianOcc 的 prob 体渲染。

nusc-sem-gs 的深度和语义图由 3D Gaussian 光栅化得到。那个 CUDA 扩展没装，
这里用同一份 occupancy_decoder 里 render_type=prob 的射线积分。
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .lift import contracted2world, world2contracted


def ray_samples(real_size, voxels_size, ratio, stepsize):
    """沿射线的采样距离。无限远收缩坐标，和官方 infinite_range 一致。"""
    xyz_min = torch.tensor(
        [real_size[0], real_size[2], real_size[4]], dtype=torch.float32)
    xyz_max = torch.tensor(
        [real_size[1], real_size[3], real_size[5]], dtype=torch.float32)
    num_voxels = float(voxels_size[0] * voxels_size[1] * voxels_size[2])
    num_id = max(int(num_voxels * (ratio ** 3)), 1)
    voxel_size = float(((xyz_max - xyz_min).prod() / num_id).pow(1.0 / 3.0))
    diagonal = float((xyz_max - xyz_min).pow(2).sum().sqrt())
    n_samples = max(int(diagonal / 2.0 / stepsize / voxel_size / ratio), 1)
    x_size = int(voxels_size[2])
    fc = 1.0 - 0.5 / x_size
    zs = torch.linspace(0.0, fc, steps=n_samples)
    roi = [-diagonal] * 3 + [diagonal] * 3
    zs_world = contracted2world(zs[:, None].expand(-1, 3).contiguous(), roi, ratio)[:, 0]
    return zs_world, voxel_size


def _camera_rays(intrinsics, cam_to_ego, height, width):
    """官方 get_rays：inverse_y，像素中心，射线不归一化。"""
    device = cam_to_ego.device
    dtype = cam_to_ego.dtype
    views = cam_to_ego.shape[0]
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    i = grid_x + 0.5
    j = grid_y + 0.5
    ones = torch.ones(views, height, width, device=device, dtype=dtype)
    dirs = torch.stack([
        (i - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None],
        (j - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None],
        ones,
    ], dim=-1)
    rot = cam_to_ego[:, None, None, :3, :3]
    rays_d = (dirs[..., None, :] * rot).sum(-1)
    rays_o = cam_to_ego[:, None, None, :3, 3].expand_as(rays_d)
    return rays_o, rays_d


def _sample_index(volume, ind):
    """ind 已经是 grid_sample 的坐标 (N, 3)。体积是 (1, C, X, Y, Z)。"""
    if ind.numel() == 0:
        return volume.new_zeros((0, volume.shape[1]))
    grid = ind.view(1, 1, 1, -1, 3).to(dtype=volume.dtype)
    sampled = F.grid_sample(volume, grid, mode='bilinear', align_corners=True)
    return sampled.reshape(volume.shape[1], -1).transpose(0, 1).contiguous()


def render_views(volume, intrinsics, cam_to_ego, pc_range, ratio, z_base, voxel_size,
                 stepsize, min_depth, semantic_sample_ratio, last_free, render_hw,
                 chunk=None, training=True):
    """volume (1, 类, X, Y, Z)，intrinsics / cam_to_ego 是 (相机数, 4, 4)。"""
    render_h, render_w = render_hw
    rays_o, rays_d = _camera_rays(intrinsics, cam_to_ego, render_h, render_w)
    flat_o = rays_o.reshape(-1, 3)
    flat_d = rays_d.reshape(-1, 3)
    z_base = z_base.to(device=volume.device, dtype=volume.dtype)
    # 一路相机 128×352。以前每 4096 条射线同步一次，一步要几百次启动。
    if chunk is None:
        chunk = render_h * render_w
    depths = []
    semantics = []
    use_ckpt = training and torch.is_grad_enabled() and volume.requires_grad
    for start in range(0, flat_o.shape[0], chunk):
        rays_o_i = flat_o[start:start + chunk]
        rays_d_i = flat_d[start:start + chunk]
        if use_ckpt:
            depth_i, sem_i = checkpoint(
                _render_chunk, volume, rays_o_i, rays_d_i, z_base,
                voxel_size, stepsize, min_depth, semantic_sample_ratio,
                last_free, list(pc_range), ratio, training,
                use_reentrant=False)
        else:
            depth_i, sem_i = _render_chunk(
                volume, rays_o_i, rays_d_i, z_base, voxel_size, stepsize,
                min_depth, semantic_sample_ratio, last_free, list(pc_range),
                ratio, training)
        depths.append(depth_i)
        semantics.append(sem_i)
    classes = volume.shape[1]
    depth = torch.cat(depths, dim=0).view(-1, 1, render_h, render_w)
    semantic = torch.cat(semantics, dim=0).view(-1, render_h, render_w, classes)
    return depth, semantic


def _render_chunk(volume, rays_o, rays_d, z_base, voxel_size, stepsize, min_depth,
                  semantic_sample_ratio, last_free, pc_range, ratio, training):
    n_rays = rays_o.shape[0]
    classes = volume.shape[1]
    # 官方 sample_ray 在 no_grad 里。深度梯度只走占据权重，不走射线坐标。
    with torch.no_grad():
        z_vals = z_base.view(1, -1).expand(n_rays, -1).clone()
        if training:
            z_vals = z_vals + (torch.rand_like(z_vals[:, :1]) * 0.2 - 0.1) * stepsize * voxel_size
            z_vals = z_vals.clamp(min=0.0)
        z_vals = z_vals + min_depth
        rays_pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
        n_samples = z_vals.shape[1]
        ind = world2contracted(rays_pts, pc_range, ratio)
        x_size, y_size, z_size = volume.shape[-3:]
        thresh = ind.new_tensor([0.4 / x_size, 0.4 / y_size, 0.4 / z_size])
        dist = torch.diff(ind, dim=1, prepend=torch.zeros_like(ind[:, :1])).abs()
        avail = (dist > thresh).any(dim=-1)
        group_size = max(int(round(1.0 / semantic_sample_ratio)), 1)
        group_size = min(group_size, n_samples)
        group_num = n_samples // group_size
        grouped = group_num * group_size
        xyz_sem = rays_pts[:, :grouped].reshape(n_rays, group_num, group_size, 3).mean(dim=2)
        avail_sem = avail[:, :grouped].reshape(n_rays, group_num, group_size).any(dim=-1)
        ind_geo = ind.flip(-1)
        ind_sem = world2contracted(xyz_sem, pc_range, ratio).flip(-1)

    geo = _sample_index(volume[:, -1:], ind_geo).view(n_rays, n_samples)
    geo_prob = torch.sigmoid(geo)
    if last_free:
        geo_prob = 1.0 - geo_prob
    last = geo_prob.new_zeros(n_rays, n_samples)
    last[:, -1] = 1.0
    weights = torch.where(avail, geo_prob, last)
    weights = weights.cumsum(dim=1).clamp(max=1.0)
    weights = weights.diff(dim=1, prepend=z_vals.new_zeros(n_rays, 1))
    depth = (weights * z_vals).sum(dim=-1)

    sem_feat = _sample_index(volume, ind_sem).view(n_rays, group_num, classes)
    sem_feat = torch.where(avail_sem.unsqueeze(-1), sem_feat, torch.zeros_like(sem_feat))
    weights_sem = weights[:, :grouped].reshape(n_rays, group_num, group_size).sum(dim=-1)
    semantic = (sem_feat * weights_sem.unsqueeze(-1)).sum(dim=-2)
    return depth, semantic
