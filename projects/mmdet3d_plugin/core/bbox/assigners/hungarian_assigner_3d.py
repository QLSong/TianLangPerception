# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
import numpy as np
import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range

    def _require_solver(self):
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

    def _pair_cost(self, bbox_pred, cls_pred, gt_bboxes, gt_labels,
                   code_weights, with_velo):
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        if code_weights is not None:
            bbox_pred = bbox_pred * code_weights
            normalized_gt_bboxes = normalized_gt_bboxes * code_weights
        if with_velo:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)
        else:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        return cls_cost + reg_cost

    def _solve(self, cost):
        """cost 是 CPU 上的 numpy 矩阵，返回 query 行号和 gt 列号。"""
        self._require_solver()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds.astype(np.int64, copy=False), matched_col_inds.astype(np.int64, copy=False)

    def _stacked_cost_ok(self):
        if getattr(self.cls_cost, 'binary_input', True):
            return False
        if not hasattr(self.cls_cost, 'eps'):
            return False
        return type(self.reg_cost).__name__ == 'BBox3DL1Cost'

    def _stacked_cost(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list,
                      counts, code_weights, with_velo):
        """一次算完所有层的代价。(层, batch, query, max_gt)，多出来的列不参与匹配。"""
        num_layers, batch_size, num_query, _ = cls_scores.shape
        max_gt = max(counts)
        pred_dim = bbox_preds.shape[-1] if with_velo else 8
        focal = self.cls_cost
        pred = cls_scores.sigmoid()
        neg_cost = -(1 - pred + focal.eps).log() * (1 - focal.alpha) * pred.pow(focal.gamma)
        pos_cost = -(pred + focal.eps).log() * focal.alpha * (1 - pred).pow(focal.gamma)
        labels = cls_scores.new_zeros((batch_size, max_gt), dtype=torch.long)
        gt_pad = bbox_preds.new_zeros((batch_size, max_gt, pred_dim))
        for img, (gt, gt_labels, num_gt) in enumerate(zip(gt_bboxes_list, gt_labels_list, counts)):
            if num_gt == 0:
                continue
            labels[img, :num_gt] = gt_labels.long()
            gt_pad[img, :num_gt] = normalize_bbox(gt, self.pc_range)[:, :pred_dim]
        picked = labels.view(1, batch_size, 1, max_gt).expand(num_layers, batch_size, num_query, max_gt)
        cls_cost = (pos_cost.gather(-1, picked) - neg_cost.gather(-1, picked)) * focal.weight
        pred_box = bbox_preds[..., :pred_dim]
        if code_weights is not None:
            pred_box = pred_box * code_weights[:pred_dim]
            gt_pad = gt_pad * code_weights[:pred_dim]
        reg_cost = torch.cdist(
            pred_box.reshape(num_layers * batch_size, num_query, pred_dim),
            gt_pad.unsqueeze(0).expand(num_layers, -1, -1, -1).reshape(
                num_layers * batch_size, max_gt, pred_dim),
            p=1).reshape(num_layers, batch_size, num_query, max_gt) * self.reg_cost.weight
        return cls_cost + reg_cost

    def match(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list,
              code_weights=None, with_velo=False):
        """所有 decoder 层、一个 batch 的匹配只把代价矩阵拷回 CPU 一次。

        cls_scores: (层, batch, query, 类别)
        bbox_preds: (层, batch, query, code)
        返回与层数等长的列表。每层是 batch 个 (rows, cols)，
        rows 是 query 下标，cols 是 gt 下标。没有 gt 时两个都为空。
        """
        if isinstance(cls_scores, (list, tuple)):
            cls_scores = torch.stack(cls_scores)
            bbox_preds = torch.stack(bbox_preds)
        num_layers, batch_size, num_query, _ = cls_scores.shape
        device = cls_scores.device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        counts = [int(gt.shape[0]) for gt in gt_bboxes_list]
        pieces = []
        valid = []
        with torch.no_grad():
            stacked = None
            if num_query > 0 and max(counts) > 0 and self._stacked_cost_ok():
                stacked = self._stacked_cost(
                    cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list,
                    counts, code_weights, with_velo)
            for layer in range(num_layers):
                for img in range(batch_size):
                    num_gt = counts[img]
                    if num_gt == 0 or num_query == 0:
                        continue
                    if stacked is None:
                        cost = self._pair_cost(
                            bbox_preds[layer, img], cls_scores[layer, img],
                            gt_bboxes_list[img], gt_labels_list[img],
                            code_weights, with_velo)
                    else:
                        cost = stacked[layer, img, :, :num_gt]
                    pieces.append(cost.reshape(-1))
                    valid.append((layer, img, num_gt))
            if pieces:
                flat = torch.cat(pieces)
                flat = torch.nan_to_num(flat, nan=100.0, posinf=100.0, neginf=-100.0)
                cpu = flat.cpu().numpy()
            else:
                cpu = None

        pairs = [[(empty, empty) for _ in range(batch_size)] for _ in range(num_layers)]
        if cpu is None:
            return pairs
        row_chunks = []
        col_chunks = []
        offset = 0
        for layer, img, num_gt in valid:
            size = num_query * num_gt
            cost = cpu[offset:offset + size].reshape(num_query, num_gt)
            offset += size
            rows, cols = self._solve(cost)
            row_chunks.append(rows)
            col_chunks.append(cols)
        rows_t = torch.from_numpy(np.concatenate(row_chunks)).to(device)
        cols_t = torch.from_numpy(np.concatenate(col_chunks)).to(device)
        start = 0
        for (layer, img, _), rows, cols in zip(valid, row_chunks, col_chunks):
            end = start + rows.shape[0]
            pairs[layer][img] = (rows_t[start:end], cols_t[start:end])
            start = end
        return pairs

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               code_weights=None,
               with_velo=False,
               eps=1e-7):
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        with torch.no_grad():
            cost = self._pair_cost(
                bbox_pred, cls_pred, gt_bboxes, gt_labels, code_weights, with_velo)
            cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
            cost = cost.cpu().numpy()
        matched_row_inds, matched_col_inds = self._solve(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)                       