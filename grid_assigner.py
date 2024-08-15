import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class GridAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposal will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum IoU for a bbox to be considered as a
            positive bbox.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=0.0,
                 gt_max_assign_all=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        super(GridAssigner, self).__init__()
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, box_responsible_flags, gt_bboxes, gt_labels=None):
        """Assign gt to bboxes. The process is very much like the max IoU
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape (n, 4).
            box_responsible_flags (Tensor): Flag to indicate whether box is
                responsible for prediction, shape (n, ).
            gt_bboxes (Tensor): Ground truth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # Compute IoU between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        # 1. Assign -1 by default
        assigned_gt_inds = paddle.full([num_bboxes], -1, dtype='int64')

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = paddle.zeros([num_bboxes])
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = paddle.full([num_bboxes], -1, dtype='int64')
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 2. Assign negative: below
        max_overlaps, argmax_overlaps = paddle.max(overlaps, axis=0)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps <= self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0]) & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. Assign positive: falls into responsible cell and above
        overlaps[:, ~box_responsible_flags.astype('bool')] = -1.0

        max_overlaps, argmax_overlaps = paddle.max(overlaps, axis=0)

        gt_max_overlaps, gt_argmax_overlaps = paddle.max(overlaps, axis=1)

        pos_inds = (max_overlaps > self.pos_iou_thr) & box_responsible_flags.astype('bool')
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. Assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & box_responsible_flags.astype('bool')
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif box_responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # Assign labels of positive anchors
        if gt_labels is not None:
            assigned_labels = paddle.full([num_bboxes], -1, dtype='int64')
            pos_inds = paddle.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
