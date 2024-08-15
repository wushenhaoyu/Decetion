import random
import numpy as np
import paddle
from paddle import tensor
from paddle.vision.transforms import functional as F
from paddleseg.utils import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

@PIPELINES.register_module()
class Expand(object):
    """Randomly expand the image and bounding boxes.
    
    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.
    
    Args:
        mean (tuple): Mean value of dataset.
        to_rgb (bool): Whether to convert the order of mean to align with RGB.
        ratio_range (tuple): Range of expand ratio.
        seg_ignore_label (int, optional): Label to ignore in segmentation masks.
        prob (float): Probability of applying this transformation.
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images and bounding boxes.
        
        Args:
            results (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Result dict with images and bounding boxes expanded.
        """
        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        
        if np.all(np.array(self.mean) == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  dtype=img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # Expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        # Expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # Expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str





import random
import numpy as np
import paddle
from paddle import tensor
from paddleseg.utils import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

def bbox_overlaps(bboxes1, bboxes2):
    """Calculate IoU between bboxes1 and bboxes2."""
    b1_x1, b1_y1, b1_x2, b1_y2 = paddle.split(bboxes1, 4, axis=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = paddle.split(bboxes2, 4, axis=1)

    inter_x1 = paddle.maximum(b1_x1, b2_x1)
    inter_y1 = paddle.maximum(b1_y1, b2_y1)
    inter_x2 = paddle.minimum(b1_x2, b2_x2)
    inter_y2 = paddle.minimum(b1_y2, b2_y2)

    inter_area = paddle.maximum(inter_x2 - inter_x1, 0) * paddle.maximum(inter_y2 - inter_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

@PIPELINES.register_module()
class MinIoURandomCrop(object):
    """Random crop the image and bounding boxes, with minimum IoU requirement.
    
    Args:
        min_ious (tuple): Minimum IoU threshold for all intersections with bounding boxes.
        min_crop_size (float): Minimum crop size ratio (i.e., h,w := a*h, a*w, where a >= min_crop_size).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU constraint.
        
        Args:
            results (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Result dict with images and bounding boxes cropped.
        """
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], 'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for _ in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue

                overlaps = bbox_overlaps(
                    paddle.to_tensor(patch.reshape(-1, 4), dtype=paddle.float32),
                    paddle.to_tensor(boxes.reshape(-1, 4), dtype=paddle.float32)
                ).numpy().reshape(-1)

                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                def is_center_of_bboxes_in_patch(boxes, patch):
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = ((centers[:, 0] > patch[0]) &
                            (centers[:, 1] > patch[1]) &
                            (centers[:, 0] < patch[2]) &
                            (centers[:, 1] < patch[3]))
                    return mask

                if len(overlaps) > 0:
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][mask.nonzero()[0]].crop(patch)

                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size})'
        return repr_str


import numpy as np
import paddle
import paddle.vision.transforms as T
from paddle.vision.transforms import functional as F

class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert isinstance(self.img_scale, list) and all(isinstance(x, tuple) for x in self.img_scale)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = 'paddle'  # PaddlePaddle backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_idx)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """
        assert len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """
        min_ratio, max_ratio = ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                results[key] = F.resize(results[key], results['scale'], interpolation='BILINEAR')
                # Update scale factors
                orig_w, orig_h = results[key].shape[1], results[key].shape[0]
                new_w, new_h = results[key].shape[2], results[key].shape[1]
                w_scale = new_w / orig_w
                h_scale = new_h / orig_h
            else:
                results[key] = F.resize(results[key], results['scale'], interpolation='BILINEAR')
                # No need for scale factors
                w_scale, h_scale = results['scale'][0] / results[key].shape[1], results['scale'][1] / results[key].shape[0]

            results['img_shape'] = results[key].shape[1:]
            results['pad_shape'] = results['img_shape']
            results['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = F.resize(results[key], results['scale'])
            else:
                results[key] = F.resize(results[key], results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                results[key] = F.resize(results[key], results['scale'], interpolation='NEAREST')
            else:
                results[key] = F.resize(results[key], results['scale'], interpolation='NEAREST')

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            assert 'scale_factor' not in results, ('scale and scale_factor cannot be both set.')

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio})'
        return repr_str




import numpy as np
import paddle
import paddle.vision.transforms as T
from paddle.vision.transforms import functional as F

class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will be
        ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert all(isinstance(x, float) for x in flip_ratio)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratio must be None, float, or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert all(isinstance(x, str) for x in direction)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes based on direction.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', 'diagonal'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, results):
        """Call function to flip images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added
                into result dict.
        """
        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [non_flip_ratio]

            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir

        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                if results['flip_direction'] == 'horizontal':
                    results[key] = F.hflip(results[key])
                elif results['flip_direction'] == 'vertical':
                    results[key] = F.vflip(results[key])
                elif results['flip_direction'] == 'diagonal':
                    results[key] = F.hflip(F.vflip(results[key]))

            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], results['img_shape'], results['flip_direction'])

            # flip masks
            for key in results.get('mask_fields', []):
                if results['flip_direction'] == 'horizontal':
                    results[key] = F.hflip(results[key])
                elif results['flip_direction'] == 'vertical':
                    results[key] = F.vflip(results[key])
                elif results['flip_direction'] == 'diagonal':
                    results[key] = F.hflip(F.vflip(results[key]))

            # flip segs
            for key in results.get('seg_fields', []):
                if results['flip_direction'] == 'horizontal':
                    results[key] = F.hflip(results[key])
                elif results['flip_direction'] == 'vertical':
                    results[key] = F.vflip(results[key])
                elif results['flip_direction'] == 'diagonal':
                    results[key] = F.hflip(F.vflip(results[key]))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio}, direction={self.direction})'



import numpy as np
import paddle
import paddle.vision.transforms as T

class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

        # Normalize transformation
        self.normalize = T.Normalize(mean=self.mean, std=self.std)

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # Convert from BGR to RGB if needed
        if self.to_rgb:
            for key in results.get('img_fields', ['img']):
                # Convert image from BGR to RGB
                results[key] = np.transpose(results[key], (2, 0, 1))  # HWC to CHW
                results[key] = results[key][::-1]  # Reverse the channels to convert BGR to RGB
        
        # Normalize image
        for key in results.get('img_fields', ['img']):
            results[key] = self.normalize(paddle.to_tensor(results[key], dtype=paddle.float32))
        
        results['img_norm_cfg'] = dict(
            mean=self.mean.tolist(), std=self.std.tolist(), to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str




import numpy as np
import paddle
import paddle.nn.functional as F

class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.size is not None:
                h, w = img.shape[:2]
                target_h, target_w = self.size
                pad_top = max(target_h - h, 0)
                pad_left = max(target_w - w, 0)
                padded_img = F.pad(
                    paddle.to_tensor(img, dtype=paddle.float32),
                    pad=[0, pad_left, 0, pad_top],
                    mode='constant',
                    value=self.pad_val
                )
                results[key] = padded_img.numpy()
            elif self.size_divisor is not None:
                h, w = img.shape[:2]
                target_h = int(np.ceil(h / self.size_divisor) * self.size_divisor)
                target_w = int(np.ceil(w / self.size_divisor) * self.size_divisor)
                pad_top = max(target_h - h, 0)
                pad_left = max(target_w - w, 0)
                padded_img = F.pad(
                    paddle.to_tensor(img, dtype=paddle.float32),
                    pad=[0, pad_left, 0, pad_top],
                    mode='constant',
                    value=self.pad_val
                )
                results[key] = padded_img.numpy()
        results['pad_shape'] = (target_h, target_w)
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape']
        for key in results.get('mask_fields', []):
            mask = results[key]
            pad_top = pad_shape[0] - mask.shape[0]
            pad_left = pad_shape[1] - mask.shape[1]
            results[key] = np.pad(mask, ((0, pad_top), (0, pad_left)), mode='constant', constant_values=self.pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            seg = results[key]
            pad_top = results['pad_shape'][0] - seg.shape[0]
            pad_left = results['pad_shape'][1] - seg.shape[1]
            results[key] = np.pad(seg, ((0, pad_top), (0, pad_left)), mode='constant', constant_values=self.pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str
