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
    for key, value in state_dict.items():
        if key not in current:
            unexpected.append(key)
            continue
        if torch.is_tensor(value) and tuple(value.shape) != tuple(current[key].shape):
            shape_skip.append(
                '{} checkpoint {} 模型 {}'.format(key, tuple(value.shape), tuple(current[key].shape)))
            continue
        loaded[key] = value
    missing = [
        key for key in current
        if key not in loaded and 'num_batches_tracked' not in key
    ]
    module.load_state_dict(loaded, strict=False)
    _log(logger, '加载预训练 {}: 写入 {} 项'.format(filename, len(loaded)))
    if unexpected:
        _log(logger, '权重里多出 {} 项，未使用'.format(len(unexpected)))
    if shape_skip:
        _log(logger, '形状不一致，跳过 {} 项: {}'.format(len(shape_skip), '; '.join(shape_skip)))
    if missing:
        _log(logger, '模型里仍用初始值的 {} 项: {}'.format(len(missing), ', '.join(missing)))
    return checkpoint
