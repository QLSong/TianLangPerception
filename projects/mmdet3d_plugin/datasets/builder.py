# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import copy
import platform
import random
from functools import partial

import cv2
import numpy as np
import torch
from mmcv.parallel import DataContainer, collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import InfiniteGroupEachSampleInBatchSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     runner_type=dict(type='EpochBasedRunner'),
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
        batch_sampler = None


    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
        batch_sampler = None


    if runner_type['type'] == 'IterBasedRunner' and shuffler_sampler['type'] =='InfiniteGroupEachSampleInBatchSampler':
        # TODO: original has more options, but I'm not using them 
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157
        batch_sampler = InfiniteGroupEachSampleInBatchSampler(
            dataset,
            samples_per_gpu,
            world_size,
            rank,
            seed=seed)
        batch_size = 1
        sampler = None


    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    # num_workers 为 0 时 PyTorch 不接受 prefetch_factor / persistent_workers。
    if num_workers > 0:
        kwargs.setdefault('prefetch_factor', 2)
        kwargs.setdefault('persistent_workers', True)
    else:
        kwargs.pop('prefetch_factor', None)
        kwargs.pop('persistent_workers', None)

    _install_container_pin_memory()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=True,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def _pin_loaded_batch(data, device=None):
    if isinstance(data, torch.Tensor):
        if data.device.type == 'cpu' and not data.is_pinned():
            return data.pin_memory(device)
        return data
    if isinstance(data, list):
        return [_pin_loaded_batch(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_pin_loaded_batch(value, device) for value in data)
    return data


def _data_container_pin_memory(container, device=None):
    """DataLoader 只会调用容器自己的 pin_memory。cpu_only 的 meta 留在页内存里。"""
    if container.cpu_only:
        return container
    return DataContainer(
        _pin_loaded_batch(container.data, device),
        container.stack,
        container.padding_value,
        cpu_only=False,
        pad_dims=container.pad_dims)


def _install_container_pin_memory():
    if getattr(DataContainer, '_tianlang_pin_memory', False):
        return
    DataContainer.pin_memory = _data_container_pin_memory
    DataContainer._tianlang_pin_memory = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    # OpenCV 默认按机器核数开线程。4 个 worker 再各开 6 路解码时会把核抢光。
    cv2.setNumThreads(1)
    if seed is None:
        return
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset