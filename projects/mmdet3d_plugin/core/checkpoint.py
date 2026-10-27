"""按名字和形状加载预训练权重。

load_from 只写入能对上的参数。优化器、迭代数留在 resume_from 里恢复。
"""
import os.path as osp
import re
from collections import OrderedDict

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner.checkpoint import _load_checkpoint


def _unwrap(model):
    while is_module_wrapper(model):
        model = model.module
    return model


def _log(logger, message):
    if logger is None:
        print(message)
    else:
        logger.info(message)


def load_pretrained(model, filename, logger=None):
    """把 checkpoint 里形状一致的参数写进模型，返回完整 checkpoint。"""
    if not filename or not osp.isfile(filename):
        raise FileNotFoundError('预训练权重不存在: {}'.format(filename))
    checkpoint = _load_checkpoint(filename, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise RuntimeError('{} 里没有 state_dict'.format(filename))
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = OrderedDict(
        (re.sub(r'^module\.', '', key), value) for key, value in state_dict.items())

    module = _unwrap(model)
    current = module.state_dict()
    loaded = OrderedDict()
    unexpected = []
    shape_skip = []
    kept_range = []
    for key, value in state_dict.items():
        if key not in current:
            unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(value.shape) != tuple(current[key].shape):
            shape_skip.append(
                '{} checkpoint {} 模型 {}'.format(key, tuple(value.shape), tuple(current[key].shape)))
            continue
        # pc_range 是坐标定义，不是学出来的权重。配置改过 z 之后仍要按配置解码。
        if key.endswith('pc_range'):
            kept_range.append((key, value.detach().float().view(-1), current[key].detach().float().view(-1)))
            continue
        loaded[key] = value
    kept_keys = {key for key, _, _ in kept_range}
    missing = [
        key for key in current
        if key not in loaded and key not in kept_keys and 'num_batches_tracked' not in key
    ]
    module.load_state_dict(loaded, strict=False)
    _log(logger, '加载预训练 {}: 写入 {} 项'.format(filename, len(loaded)))
    if unexpected:
        _log(logger, '权重里多出 {} 项，未使用'.format(len(unexpected)))
    if shape_skip:
        _log(logger, '形状不一致，跳过 {} 项: {}'.format(len(shape_skip), '; '.join(shape_skip)))
    if missing:
        _log(logger, '模型里仍用初始值的 {} 项: {}'.format(len(missing), ', '.join(missing)))
    for key, old, new in kept_range:
        shift = float(new[2] - old[2]) if old.numel() >= 6 and abs(float((new[5] - new[2]) - (old[5] - old[2]))) < 1e-4 else None
        message = '保留配置 {} {}，权重里是 {}'.format(
            key, [round(float(v), 2) for v in new], [round(float(v), 2) for v in old])
        if shift is not None:
            message += '，z 跨度不变，解码整体 {:+.2f} m'.format(shift)
        _log(logger, message)
    return checkpoint
