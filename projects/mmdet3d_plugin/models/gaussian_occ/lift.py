import torch
import torch.nn.functional as F


def apply_4x4(rt, xyz):
    ones = torch.ones_like(xyz[..., :1])
    xyz1 = torch.cat([xyz, ones], dim=-1)
    return torch.matmul(rt, xyz1.transpose(-1, -2)).transpose(-1, -2)[..., :3]


def scale_intrinsics(k, sx, sy):
    scaled = k.clone()
    scaled[:, 0, 0] = k[:, 0, 0] * sx
    scaled[:, 0, 2] = k[:, 0, 2] * sx
    scaled[:, 1, 1] = k[:, 1, 1] * sy
    scaled[:, 1, 2] = k[:, 1, 2] * sy
    return scaled


def world2contracted(xyz, pc_range, ratio):
    xyz_min = xyz.new_tensor(pc_range[:3])
    xyz_max = xyz.new_tensor(pc_range[3:])
    t = ratio / (1.0 - ratio)
    xyz_scaled = (2 * (xyz - xyz_min) / (xyz_max - xyz_min) - 1) * t
    xyz_abs = xyz_scaled.abs()
    xyz_contracted = torch.where(
        xyz_abs <= t,
        xyz_scaled,
        xyz_scaled.sign() * (1.0 + t - 1.0 / (xyz_abs + 1 - t)))
    return xyz_contracted / (t + 1)


def contracted2world(xyz, pc_range, ratio):
    xyz_min = xyz.new_tensor(pc_range[:3])
    xyz_max = xyz.new_tensor(pc_range[3:])
    t = ratio / (1.0 - ratio)
    xyz_ = xyz * (t + 1)
    xyz_abs = xyz_.abs()
    xyz_scaled = torch.where(
        xyz_abs <= t,
        xyz_,
        xyz_.sign() * (t - 1.0 + 1.0 / (t + 1 - xyz_abs))) / t
    return 0.5 * (xyz_scaled + 1) * (xyz_max - xyz_min) + xyz_min


def gridcloud3d(batch, depth, height, width, device):
    grid_z = torch.linspace(0, depth - 1, depth, device=device)
    grid_y = torch.linspace(0, height - 1, height, device=device)
    grid_x = torch.linspace(0, width - 1, width, device=device)
    z, y, x = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
    xyz = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
    return xyz.expand(batch, -1, -1).contiguous()


def mem_to_world(xyz_mem, size_xyz, pc_range, ratio):
    x_size, y_size, z_size = size_xyz
    xyz = xyz_mem.clone()
    xyz[..., 0] = xyz[..., 0].clamp(0.5, x_size - 1.5)
    xyz[..., 1] = xyz[..., 1].clamp(0.5, y_size - 1.5)
    xyz[..., 2] = xyz[..., 2].clamp(0.05, z_size - 1.05)
    denom = xyz.new_tensor([x_size - 1, y_size - 1, z_size - 1])
    xyz_norm = xyz / denom * 2 - 1
    return contracted2world(xyz_norm, pc_range, ratio)


def _lift_views(feat, pix, cam, xyz_world, height, width, depth, y_size, x_size):
    """一帧的全部相机一起投到体素。feat 是 (V, C, H, W)，返回 (C, Z, Y, X)。"""
    views = feat.shape[0]
    xyz_cam = apply_4x4(cam, xyz_world)
    xyz_pix = apply_4x4(pix, xyz_world)
    depth_cam = xyz_cam[..., 2]
    xy = xyz_pix[..., :2] / xyz_pix[..., 2:3].clamp(min=1e-6)
    # 坐标先用 float32 算完，再转成和特征相同的精度。这一版 grid_sample 不接受混用。
    sample_feat = feat.half() if feat.is_cuda and feat.dtype == torch.float32 else feat
    valid = (
        (xy[..., 0] > -0.5) & (xy[..., 0] < width - 0.5)
        & (xy[..., 1] > -0.5) & (xy[..., 1] < height - 0.5)
        & (depth_cam > 0)).to(dtype=sample_feat.dtype).view(views, 1, depth, y_size, x_size)
    x_norm = 2.0 * (xy[..., 0] / float(width - 1)) - 1.0
    y_norm = 2.0 * (xy[..., 1] / float(height - 1)) - 1.0
    grid = torch.stack([x_norm, y_norm, torch.zeros_like(x_norm)], dim=-1)
    grid = grid.view(views, depth, y_size, x_size, 3).to(dtype=sample_feat.dtype)
    sampled = F.grid_sample(sample_feat.unsqueeze(2), grid, align_corners=False)
    sampled = sampled * valid
    hit = (sampled.abs() > 0).to(dtype=sampled.dtype)
    return sampled.sum(dim=0), hit.sum(dim=0)


def unproject_feature(features, ego_to_pix, ego_to_cam, size_zyx, pc_range, ratio):
    """把多相机特征抬到收缩坐标体素。返回 (B, C, X, Y, Z)。

    调用方已经按样本切开，这里把该样本的六路相机一次采样完。
    """
    batch, views, channels, height, width = features.shape
    depth, y_size, x_size = size_zyx
    pix = ego_to_pix.reshape(batch, views, 4, 4)
    cam = ego_to_cam.reshape(batch, views, 4, 4)
    xyz_world = mem_to_world(
        gridcloud3d(1, depth, y_size, x_size, features.device),
        (x_size, y_size, depth), pc_range, ratio)
    numer, denom = [], []
    for index in range(batch):
        sampled, hit = _lift_views(
            features[index], pix[index], cam[index], xyz_world,
            height, width, depth, y_size, x_size)
        numer.append(sampled)
        denom.append(hit)
    feat = torch.stack(numer, dim=0) / torch.stack(denom, dim=0).clamp(min=1e-6)
    return feat.permute(0, 1, 4, 3, 2).contiguous()


def sample_occ(volume, pc_range, ratio, volume_size, occ_size=(200, 200, 16), occ_range=None):
    """把体素特征采样到 Occ3D 的 200×200×16。volume 是 (B, C, X, Y, Z)。"""
    if occ_range is None:
        occ_range = pc_range
    batch = volume.shape[0]
    x_size, y_size, z_size = occ_size
    xyz = gridcloud3d(batch, z_size, y_size, x_size, volume.device)
    occ_min = volume.new_tensor(occ_range[:3])
    occ_max = volume.new_tensor(occ_range[3:])
    span = occ_max - occ_min
    num_voxels = float(volume_size[0] * volume_size[1] * volume_size[2])
    num_id = max(int(num_voxels * (ratio ** 3)), 1)
    voxel = (span.prod() / num_id).pow(1.0 / 3.0)
    occ_num = volume.new_tensor([float(x_size), float(y_size), float(z_size)])
    xyz = xyz / occ_num * span + occ_min + 0.5 * voxel
    ind = world2contracted(xyz, pc_range, ratio).flip(-1)
    ind = ind.view(batch, 1, 1, -1, 3).to(dtype=volume.dtype)
    sampled = F.grid_sample(volume, ind, mode='bilinear', align_corners=True)
    sampled = sampled.squeeze(2).squeeze(2).transpose(1, 2)
    return sampled.reshape(batch, z_size, y_size, x_size, -1).permute(0, 4, 3, 2, 1).contiguous()
