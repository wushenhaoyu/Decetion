import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D
from paddle.nn.initializer import KaimingUniform, Constant

# 自定义实现的模块，需要根据实际情况添加
def ConvModule(in_channels, out_channels, kernel_size, stride=1, padding=0, 
                norm_cfg=None, act_cfg=None):
    layers = []
    layers.append(Conv2D(in_channels, out_channels, kernel_size, stride, padding))
    if norm_cfg:
        norm_layer = build_norm_layer(norm_cfg, out_channels)
        layers.append(norm_layer)
    if act_cfg:
        activation = build_activation_layer(act_cfg)
        layers.append(activation)
    return nn.Sequential(*layers)

def normal_init(layer, mean=0.0, std=1.0):
    """Initialize weights with normal distribution."""
    if isinstance(layer, nn.Conv2D):
        layer.weight.set_value(paddle.normal(mean=mean, std=std, shape=layer.weight.shape))
        if layer.bias is not None:
            layer.bias.set_value(paddle.normal(mean=mean, std=std, shape=layer.bias.shape))

def force_fp32(tensor):
    """Force the tensor to float32."""
    return paddle.cast(tensor, dtype='float32')

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

def build_norm_layer(cfg, num_features):
    """Build normalization layer."""
    norm_type = cfg.get('type', 'BN')
    if norm_type == 'BN':
        return nn.BatchNorm2D(num_features)
    else:
        raise ValueError(f'Unsupported normalization type: {norm_type}')

def build_activation_layer(cfg):
    """Build activation layer."""
    act_type = cfg.get('type', 'ReLU')
    if act_type == 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(f'Unsupported activation type: {act_type}')

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

@HEADS.register_module()
class YOLOV3Head(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV3Head, self).__init__()
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_xy = build_loss(loss_xy)
        self.loss_wh = build_loss(loss_wh)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        self.convs_bridge = nn.LayerList()
        self.convs_pred = nn.LayerList()
        for i in range(self.num_levels):
            conv_bridge = nn.Sequential(
                nn.Conv2D(self.in_channels[i], self.out_channels[i], 3, padding=1),
                nn.BatchNorm2D(self.out_channels[i]),
                nn.LeakyReLU(negative_slope=0.1)
            )
            conv_pred = nn.Conv2D(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            nn.initializer.Normal(m.weight, mean=0.0, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (paddle.nn.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (paddle.nn.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The same as get_bboxes.
        """
        num_classes = self.num_classes
        result_list = []

        anchors, base_anchors = self.anchor_generator.grid_anchors(
            self.featmap_strides)
        for i in range(len(pred_maps_list)):
            pred_map = pred_maps_list[i]
            featmap_stride = self.featmap_strides[i]
            base_anchor = base_anchors[i]

            assert (base_anchor.size(0) == self.num_anchors)
            pred_map = pred_map.permute(0, 2, 3, 1).reshape(
                [pred_map.shape[0], pred_map.shape[2], pred_map.shape[3],
                 self.num_anchors, -1])

            # decode prediction
            bboxes = self.bbox_coder.decode(
                pred_map[..., :4].reshape(
                    [pred_map.shape[0], pred_map.shape[1], pred_map.shape[2],
                     -1]),
                anchors[i], base_anchor, featmap_stride)
            scores = pred_map[..., 4].sigmoid()
            cls_scores = pred_map[..., 5:].sigmoid()
            cls_scores = F.softmax(cls_scores, axis=-1)

            # compute bbox scores
            bbox_scores = scores.unsqueeze(3) * cls_scores
            bboxes = bboxes.reshape([bboxes.shape[0], -1, 4])
            bbox_scores = bbox_scores.reshape([bbox_scores.shape[0], -1, num_classes])
            bboxes, bbox_scores = multi_apply(
                self._get_bboxes_single,
                bboxes,
                bbox_scores,
                cfg,
                rescale=rescale,
                with_nms=with_nms)

            result_list.append((bboxes, bbox_scores))
        return result_list

    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg=None,
             reduction='mean'):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): List of raw prediction tensors with the
                same number of elements as the `pred_maps` in forward().
            gt_bboxes (list[Tensor]): List of ground truth bboxes with the same
                number of elements as `pred_maps`.
            gt_labels (list[Tensor]): List of ground truth labels with the same
                number of elements as `pred_maps`.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (paddle.nn.Config | None): Training configuration for losses
                and post-processing. Default: None.
            reduction (str): The method to reduce the loss. Options are 'none',
                'mean', and 'sum'. Default: 'mean'.

        Returns:
            dict: A dictionary containing the loss values.
        """
        assert len(pred_maps) == len(gt_bboxes) == len(gt_labels)
        losses = dict()

        if self.train_cfg:
            assign_result = []
            sampling_result = []
            for i in range(len(gt_bboxes)):
                assign_result.append(self.assigner.assign(
                    gt_bboxes[i], gt_labels[i], self.anchor_generator.grid_anchors(
                        [self.featmap_strides[i]] * len(gt_bboxes))[i]))
                sampling_result.append(self.sampler.sample(assign_result[i],
                                                           gt_bboxes[i],
                                                           gt_labels[i]))
            sampling_result = list(zip(*sampling_result))
            anchors = self.anchor_generator.grid_anchors(self.featmap_strides)

            # compute loss
            loss_cls, loss_conf, loss_xy, loss_wh = multi_apply(
                self._loss_single,
                pred_maps,
                sampling_result[0],
                sampling_result[1],
                anchors)

            losses['loss_cls'] = sum(loss_cls) / len(loss_cls)
            losses['loss_conf'] = sum(loss_conf) / len(loss_conf)
            losses['loss_xy'] = sum(loss_xy) / len(loss_xy)
            losses['loss_wh'] = sum(loss_wh) / len(loss_wh)

        return losses

    def _loss_single(self,
                     pred_map,
                     anchor,
                     sampling_result,
                     anchors):
        """Calculate loss for a single scale.

        Args:
            pred_map (Tensor): Raw prediction map.
            anchor (Tensor): Anchor.
            sampling_result (Tensor): Sampling result.
            anchors (Tensor): Anchors.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: Loss values for classification,
                confidence, xy-coordinates and width-height.
        """
        # decoding the predicted bbox
        pred_map = pred_map.permute(0, 2, 3, 1).reshape(
            [pred_map.shape[0], pred_map.shape[2], pred_map.shape[3], self.num_anchors, -1])
        pred_bboxes = pred_map[..., :4].reshape(
            [pred_map.shape[0], pred_map.shape[1], pred_map.shape[2], -1])
        pred_scores = pred_map[..., 4].sigmoid()
        pred_cls = pred_map[..., 5:].sigmoid()

        # computing loss
        anchor, pos_indices, neg_indices = anchor
        gt_bboxes, gt_labels = sampling_result

        loss_cls = self.loss_cls(pred_cls, gt_labels)
        loss_conf = self.loss_conf(pred_scores, pos_indices)
        loss_xy = self.loss_xy(pred_bboxes, gt_bboxes)
        loss_wh = self.loss_wh(pred_bboxes, gt_bboxes)

        return loss_cls, loss_conf, loss_xy, loss_wh
    import paddle
    import paddle.nn.functional as F
    from ppdet.core import images_to_levels, multi_apply

    def get_targets(self, anchor_list, responsible_flag_list, gt_bboxes_list, gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi-level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi-level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(anchor_list)

        # Anchor number of multi-levels
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        
        results = multi_apply(self._get_targets_single, anchor_list,
                            responsible_flag_list, gt_bboxes_list,
                            gt_labels_list)

        all_target_maps, all_neg_maps = results
        assert num_imgs == len(all_target_maps) == len(all_neg_maps)
        target_maps_list = images_to_levels(all_target_maps, num_level_anchors)
        neg_maps_list = images_to_levels(all_neg_maps, num_level_anchors)

        return target_maps_list, neg_maps_list

    def _get_targets_single(self, anchors, responsible_flags, gt_bboxes, gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors.
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Prediction target map of each
                    scale level, shape (num_total_anchors, 5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """
        anchor_strides = [paddle.full(shape=[len(anchors[i])], dtype=anchors[i].dtype, fill_value=self.featmap_strides[i]) for i in range(len(anchors))]
        concat_anchors = paddle.concat(anchors, axis=0)
        concat_responsible_flags = paddle.concat(responsible_flags, axis=0)
        anchor_strides = paddle.concat(anchor_strides, axis=0)
        
        assert len(anchor_strides) == len(concat_anchors) == len(concat_responsible_flags)
        
        assign_result = self.assigner.assign(concat_anchors, concat_responsible_flags, gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, concat_anchors, gt_bboxes)
        
        target_map = paddle.zeros([concat_anchors.shape[0], self.num_attrib], dtype=concat_anchors.dtype)
        
        target_map.index_set_(sampling_result.pos_inds, paddle.concat([
            self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes, anchor_strides[sampling_result.pos_inds]).unsqueeze(1),
            paddle.full([sampling_result.pos_inds.shape[0], 1], 1, dtype=paddle.float32)
        ], axis=1))
        
        gt_labels_one_hot = F.one_hot(gt_labels, num_classes=self.num_classes).astype(paddle.float32)
        if self.one_hot_smoother != 0:  # label smooth
            gt_labels_one_hot = gt_labels_one_hot * (1 - self.one_hot_smoother) + self.one_hot_smoother / self.num_classes
        target_map.index_set_(sampling_result.pos_inds, gt_labels_one_hot[sampling_result.pos_assigned_gt_inds].unsqueeze(1))

        neg_map = paddle.zeros([concat_anchors.shape[0]], dtype=paddle.uint8)
        neg_map.index_set_(sampling_result.neg_inds, paddle.full([sampling_result.neg_inds.shape[0]], 1, dtype=paddle.uint8))

        return target_map, neg_map

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): The outer list indicates test-time
                augmentations and the inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): The outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. Each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: BBox results of each class.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
