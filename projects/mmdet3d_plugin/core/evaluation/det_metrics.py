"""自车坐标系上的 nuScenes 检测指标。

框已经在自车系，中心距和官方协议一样。mAP 对 0.5/1/2/4 米取平均，
precision、recall 是分数阈值处、中心距 2 米的工作点。
没有属性标注，mAAE 不计入 NDS。
"""
import numpy as np
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionBox
from pyquaternion import Quaternion

CLASS_RANGE = {
    'car': 50, 'truck': 50, 'bus': 50, 'trailer': 50,
    'construction_vehicle': 50, 'pedestrian': 40, 'motorcycle': 40,
    'bicycle': 40, 'traffic_cone': 30, 'barrier': 30,
}
DIST_THS = (0.5, 1.0, 2.0, 4.0)
DIST_TH_TP = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MAX_BOXES = 500
TP_NAMES = ('trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err')
TP_PRETTY = {
    'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE',
    'vel_err': 'mAVE', 'attr_err': 'mAAE',
}


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pack_boxes(boxes, labels, scores=None):
    """把 LiDARInstance3DBoxes 收成 numpy。空框返回长度为 0 的数组。"""
    labels = np.zeros((0,), np.int64) if labels is None else _numpy(labels).reshape(-1)
    if boxes is None or len(boxes) == 0:
        empty = np.zeros((0, 3), np.float32)
        packed = dict(center=empty, size=empty, yaw=np.zeros((0,), np.float32),
                      vel=np.zeros((0, 2), np.float32), label=labels.astype(np.int64))
        if scores is not None:
            packed['score'] = np.zeros((0,), np.float32)
        return packed
    center = boxes.gravity_center.detach().cpu().numpy().astype(np.float32)
    dims = boxes.dims.detach().cpu().numpy().astype(np.float32)
    size = dims[:, [1, 0, 2]]
    yaw = boxes.yaw.detach().cpu().numpy().astype(np.float32)
    tensor = boxes.tensor.detach().cpu().numpy()
    vel = tensor[:, 7:9].astype(np.float32) if tensor.shape[1] > 8 else np.zeros((len(boxes), 2), np.float32)
    packed = dict(center=center, size=size, yaw=yaw, vel=vel, label=np.asarray(labels).astype(np.int64))
    if scores is not None:
        packed['score'] = _numpy(scores).reshape(-1).astype(np.float32)
    return packed


def _keep_top(pred):
    score = pred['score']
    if len(score) <= MAX_BOXES:
        return pred
    order = np.argsort(-score)[:MAX_BOXES]
    return {key: value[order] for key, value in pred.items()}


def _in_range(center, name):
    limit = CLASS_RANGE.get(name)
    if limit is None:
        return False
    return float(np.linalg.norm(center[:2])) < limit


def _to_boxes(token, packed, class_names, score_thr=None):
    boxes = []
    names = list(class_names)
    n = len(packed['label'])
    for i in range(n):
        label = int(packed['label'][i])
        if label < 0 or label >= len(names):
            continue
        name = names[label]
        if not _in_range(packed['center'][i], name):
            continue
        score = None if 'score' not in packed else float(packed['score'][i])
        if score_thr is not None and score < score_thr:
            continue
        quat = Quaternion(axis=[0, 0, 1], radians=float(packed['yaw'][i]))
        boxes.append(DetectionBox(
            sample_token=token,
            translation=tuple(float(v) for v in packed['center'][i]),
            size=tuple(float(v) for v in packed['size'][i]),
            rotation=tuple(float(v) for v in quat.elements),
            velocity=(float(packed['vel'][i, 0]), float(packed['vel'][i, 1])),
            ego_translation=tuple(float(v) for v in packed['center'][i]),
            num_pts=-1,
            detection_name=name,
            detection_score=-1.0 if score is None else score,
        ))
    return boxes


def _operating_point(records, class_names, score_thr):
    """分数不低于阈值、中心距小于 2 米时的 precision / recall。"""
    names = [name for name in class_names if name in CLASS_RANGE]
    tp = {name: 0 for name in names}
    fp = {name: 0 for name in names}
    fn = {name: 0 for name in names}
    for record in records:
        pred = _keep_top(record['pred'])
        gt_boxes = _to_boxes(record['token'], record['gt'], names)
        pred_boxes = _to_boxes(record['token'], pred, names, score_thr=score_thr)
        by_name_gt = {}
        for box in gt_boxes:
            by_name_gt.setdefault(box.detection_name, []).append(box)
        by_name_pred = {}
        for box in pred_boxes:
            by_name_pred.setdefault(box.detection_name, []).append(box)
        for name in names:
            gts = by_name_gt.get(name, [])
            preds = sorted(by_name_pred.get(name, []), key=lambda box: -box.detection_score)
            taken = set()
            hit = 0
            for pred_box in preds:
                best, best_dist = None, np.inf
                for index, gt_box in enumerate(gts):
                    if index in taken:
                        continue
                    dist = center_distance(gt_box, pred_box)
                    if dist < best_dist:
                        best, best_dist = index, dist
                if best is not None and best_dist < DIST_TH_TP:
                    taken.add(best)
                    hit += 1
            tp[name] += hit
            fp[name] += len(preds) - hit
            fn[name] += len(gts) - hit
    precision, recall = {}, {}
    for name in names:
        precision[name] = tp[name] / (tp[name] + fp[name]) if tp[name] + fp[name] else 0.0
        recall[name] = tp[name] / (tp[name] + fn[name]) if tp[name] + fn[name] else 0.0
    return precision, recall


def evaluate_det(records, class_names, score_thr=0.3):
    """records 是 pack 过的逐帧检测。返回可打印的文本和指标字典。"""
    names = [name for name in class_names if name in CLASS_RANGE]
    gt_boxes, pred_boxes = EvalBoxes(), EvalBoxes()
    for record in records:
        token = record['token']
        gt_boxes.add_boxes(token, _to_boxes(token, record['gt'], names))
        pred_boxes.add_boxes(token, _to_boxes(token, _keep_top(record['pred']), names))

    ap = {name: {} for name in names}
    tp = {name: {} for name in names}
    for name in names:
        for dist in DIST_THS:
            metric = accumulate(gt_boxes, pred_boxes, name, center_distance, dist)
            ap[name][dist] = calc_ap(metric, MIN_RECALL, MIN_PRECISION)
        metric = accumulate(gt_boxes, pred_boxes, name, center_distance, DIST_TH_TP)
        for metric_name in TP_NAMES:
            if metric_name == 'attr_err':
                tp[name][metric_name] = np.nan
            elif name == 'traffic_cone' and metric_name in ('vel_err', 'orient_err'):
                tp[name][metric_name] = np.nan
            elif name == 'barrier' and metric_name == 'vel_err':
                tp[name][metric_name] = np.nan
            else:
                tp[name][metric_name] = calc_tp(metric, MIN_RECALL, metric_name)

    mean_ap = {name: float(np.mean(list(values.values()))) for name, values in ap.items()}
    map_score = float(np.mean(list(mean_ap.values()))) if mean_ap else 0.0
    tp_mean = {}
    for metric_name in TP_NAMES:
        values = [tp[name][metric_name] for name in names]
        tp_mean[metric_name] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else np.nan
    scores = [max(0.0, 1.0 - err) for err in tp_mean.values() if not np.isnan(err)]
    nds = (5.0 * map_score + sum(scores)) / (5.0 + len(scores)) if scores else map_score
    precision, recall = _operating_point(records, names, score_thr)

    lines = [
        'det  precision/recall @ score>={:.2f}, center dist<{:.1f}m'.format(score_thr, DIST_TH_TP),
        '{:<22} {:>8} {:>8} {:>8}'.format('class', 'P', 'R', 'AP'),
    ]
    for name in names:
        lines.append('{:<22} {:8.3f} {:8.3f} {:8.3f}'.format(
            name, precision[name], recall[name], mean_ap[name]))
    lines.append('{:<22} {:8.3f} {:8.3f} {:8.3f}'.format(
        'MEAN',
        float(np.mean(list(precision.values()))) if precision else 0.0,
        float(np.mean(list(recall.values()))) if recall else 0.0,
        map_score))
    detail = '  '.join(
        '{} {:.3f}'.format(TP_PRETTY[name], tp_mean[name])
        for name in TP_NAMES if not np.isnan(tp_mean[name]))
    lines.append('mAP {:.3f}  NDS {:.3f}  {}'.format(map_score, nds, detail))
    lines.append('mAAE 没有属性标注，不计入 NDS')
    text = '\n'.join(lines)
    return dict(mAP=map_score, NDS=nds, precision=precision, recall=recall, AP=mean_ap), text
