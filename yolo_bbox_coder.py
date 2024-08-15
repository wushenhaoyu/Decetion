import paddle
import numpy as np
from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder

@BBOX_CODERS.register_module()
class YOLOBBoxCoder(BaseBBoxCoder):
    """YOLO BBox coder.

    Following `YOLO <https://arxiv.org/abs/1506.02640>`_, this coder divides
    the image into grids, and encodes bbox (x1, y1, x2, y2) into (cx, cy, dw, dh).
    cx, cy in [0., 1.], denotes relative center position w.r.t the center of
    bboxes. dw, dh are the same as :obj:`DeltaXYWHBBoxCoder`.

    Args:
        eps (float): Min value of cx, cy when encoding.
    """

    def __init__(self, eps=1e-6):
        super(YOLOBBoxCoder, self).__init__()
        self.eps = eps

    def encode(self, bboxes, gt_bboxes, stride):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (paddle.Tensor): Source boxes, e.g., anchors.
            gt_bboxes (paddle.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
            stride (paddle.Tensor | int): Stride of bboxes.

        Returns:
            paddle.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] == gt_bboxes.shape[-1] == 4
        x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
        y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
        h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        w_target = paddle.log((w_gt / w).clip(min=self.eps))
        h_target = paddle.log((h_gt / h).clip(min=self.eps))
        x_center_target = ((x_center_gt - x_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        y_center_target = ((y_center_gt - y_center) / stride + 0.5).clip(
            self.eps, 1 - self.eps)
        encoded_bboxes = paddle.stack(
            [x_center_target, y_center_target, w_target, h_target], axis=-1)
        return encoded_bboxes

    def decode(self, bboxes, pred_bboxes, stride):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (paddle.Tensor): Basic boxes, e.g. anchors.
            pred_bboxes (paddle.Tensor): Encoded boxes with shape
            stride (paddle.Tensor | int): Strides of bboxes.

        Returns:
            paddle.Tensor: Decoded boxes.
        """
        assert pred_bboxes.shape[0] == bboxes.shape[0]
        assert pred_bboxes.shape[-1] == bboxes.shape[-1] == 4
        x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0]
        h = bboxes[..., 3] - bboxes[..., 1]
        # Get outputs x, y
        x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
        y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
        w_pred = paddle.exp(pred_bboxes[..., 2]) * w
        h_pred = paddle.exp(pred_bboxes[..., 3]) * h

        decoded_bboxes = paddle.stack(
            (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
             x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
            axis=-1)

        return decoded_bboxes
