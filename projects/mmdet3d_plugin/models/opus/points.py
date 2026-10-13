import torch


def encode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    points[..., 1] = (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    points[..., 2] = (points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return points


def decode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points
