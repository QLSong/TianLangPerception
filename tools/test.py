# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
'''
python tools/test.py projects/configs/TianLangEyes/streampetr_convnext_pp_lss_nus.py CKPT \
  --launcher pytorch --eval det occ \
  --vis-dir work_dirs/vis_val --vis-interval 50 --score-thr 0.3
'''
import argparse
import math
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import mmcv
import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.core.apis.test import collect_results_cpu
from projects.mmdet3d_plugin.core.evaluation.det_metrics import evaluate_det, pack_boxes
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import OccMeter, crop_occ
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import lidar_to_ego_matrix
from projects.mmdet3d_plugin.datasets.pipelines.loading import _index_occ_root
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='precision/recall 和渲染用的分数阈值')
    parser.add_argument('--vis-dir', help='抽帧渲染的输出目录')
    parser.add_argument('--vis-interval', type=int, default=0,
                        help='每隔多少帧渲染一张，0 表示不渲染')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def task_names(cfg):
    tasks = cfg.model.get('tasks', 'det')
    if isinstance(tasks, str):
        return ('det', 'occ') if tasks == 'both' else (tasks,)
    return tuple(tasks)


def eval_tasks(requested, tasks):
    if not requested:
        return set()
    wanted = set()
    for name in requested:
        if name in ('det', 'bbox', 'mAP'):
            wanted.add('det')
        elif name in ('occ', 'miou', 'mIoU', 'rayiou', 'RayIoU'):
            wanted.add('occ')
    return wanted & set(tasks)


def sampler_indices(n, rank, world_size):
    """和 DistributedSampler(shuffle=False) 相同的下标，含末尾补齐。"""
    if n == 0:
        return []
    total = math.ceil(n / world_size) * world_size
    indices = (list(range(n)) * math.ceil(total / n))[:total]
    per = total // world_size
    return indices[rank * per:(rank + 1) * per]


def reduce_occ(meter):
    if not (dist.is_available() and dist.is_initialized()):
        return
    parts = (meter.hist, meter.gt_cnt, meter.pred_cnt, meter.tp_cnt)
    flat = np.concatenate([part.reshape(-1) for part in parts])
    tensor = torch.tensor(flat, dtype=torch.float64, device='cuda')
    dist.all_reduce(tensor)
    values = tensor.cpu().numpy()
    offset = 0
    for part in parts:
        size = part.size
        part[:] = values[offset:offset + size].reshape(part.shape)
        offset += size


def take_pred_boxes(result, score_thr):
    pred = result.get('pts_bbox')
    if pred is None:
        return None, None, None
    keep = pred['scores_3d'] >= score_thr
    return pred['boxes_3d'][keep], pred['labels_3d'][keep], pred['scores_3d'][keep]


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.vis_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--vis-dir"')
    if args.vis_dir and args.vis_interval <= 0:
        raise ValueError('--vis-dir 需要 --vis-interval 大于 0')
    if args.format_only:
        raise ValueError('检测框在自车系，不再导出 nuScenes json。指标用 --eval det 或 --eval occ')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # 时序记忆按帧更新，测试固定每次一个样本。
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
            ds_cfg.pop('samples_per_gpu', None)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'pytorch' and 'RANK' not in os.environ:
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29501')
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    tasks = task_names(cfg)
    metrics = eval_tasks(args.eval, tasks)
    show_det = 'det' in tasks
    show_occ = 'occ' in tasks
    rank, world_size = get_dist_info()
    if rank == 0 and args.eval:
        skipped = []
        for name in args.eval:
            kind = 'det' if name in ('det', 'bbox', 'mAP') else 'occ' if name in (
                'occ', 'miou', 'mIoU', 'rayiou', 'RayIoU') else None
            if kind and kind not in tasks:
                skipped.append(kind)
        if skipped:
            print('配置 task={}，跳过 {}'.format(cfg.model.get('tasks'), '、'.join(skipped)))
    indices = sampler_indices(len(dataset), rank, world_size)
    det_records = []
    occ_meter = OccMeter() if 'occ' in metrics else None
    occ_index = _index_occ_root(cfg.occ_root) if (show_occ and (occ_meter or args.vis_dir)) else {}
    label_range = list(cfg.model.occ_head.get(
        'occ_label_range', [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]))
    grid = cfg.model.img_view_transformer.grid_config if show_occ else None
    if args.vis_dir and rank == 0:
        os.makedirs(args.vis_dir, exist_ok=True)
    if distributed:
        dist.barrier()
    if args.vis_dir:
        import vis_mini
        pc = cfg.point_cloud_range
        det_bounds = (pc[0], pc[1], pc[3], pc[4])
        z_bounds = (label_range[2], label_range[5])
        class_names = list(cfg.class_names)
    model.eval()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for step, data in enumerate(data_loader):
        if step >= len(indices):
            break
        index = indices[step]
        if rank * len(indices) + step >= len(dataset):
            continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)[0]
        info = dataset.data_infos[index]
        if 'det' in metrics and result.get('pts_bbox') is not None:
            ann = dataset.get_ann_info(index)
            pred = result['pts_bbox']
            det_records.append(dict(
                token=info['token'],
                gt=pack_boxes(ann['gt_bboxes_3d'], ann['gt_labels_3d']),
                pred=pack_boxes(pred['boxes_3d'], pred['labels_3d'], pred['scores_3d']),
            ))
        occ_gt = occ_mask = None
        if show_occ and (occ_meter is not None or (args.vis_dir and index % args.vis_interval == 0)):
            gt_path = occ_index.get(info['token'])
            if gt_path is None:
                raise FileNotFoundError('没有找到 {} 的占用标签'.format(info['token']))
            occ_file = np.load(gt_path)
            occ_gt = occ_file['semantics']
            occ_mask = occ_file['mask_camera'].astype(bool)
        if occ_meter is not None:
            pred_occ = crop_occ(result['occ'], grid, label_range)
            origin = lidar_to_ego_matrix(info)[:3, 3]
            occ_meter.update(pred_occ, occ_gt, occ_mask, origin)
        if args.vis_dir and index % args.vis_interval == 0:
            gt_boxes = gt_labels = None
            pred_boxes = pred_labels = pred_scores = None
            if show_det:
                ann = dataset.get_ann_info(index)
                gt_boxes, gt_labels = ann['gt_bboxes_3d'], ann['gt_labels_3d']
                pred_boxes, pred_labels, pred_scores = take_pred_boxes(result, args.score_thr)
            image = vis_mini.render_frame(
                info, lidar_to_ego_matrix(info), gt_boxes, gt_labels,
                pred_boxes, pred_labels, pred_scores, occ_gt, result.get('occ'),
                show_det, show_occ, det_bounds, label_range, grid, class_names, z_bounds)
            name = '{:03d}_{}.jpg'.format(index, info['token'][:8])
            vis_mini.cv2.imwrite(osp.join(args.vis_dir, name), image)
        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()

    if distributed:
        det_records = collect_results_cpu(det_records, len(dataset), args.tmpdir)
        if occ_meter is not None:
            reduce_occ(occ_meter)
    if rank == 0 and args.out and det_records:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(det_records, args.out)
    if rank == 0 and 'det' in metrics:
        _, text = evaluate_det(det_records or [], list(cfg.class_names), args.score_thr)
        print('\n' + text)
    if rank == 0 and occ_meter is not None:
        _, text = occ_meter.summary()
        print('\n' + text)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
