import warnings
import numpy as np

from paddle import to_tensor
from paddle.vision.transforms import Compose, Resize, RandomHorizontalFlip, Normalize, Pad
from paddle.vision.transforms import ToTensor
from ..builder import PIPELINES

@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float | list[float] | None): Scale factors for resizing.
        flip (bool): Whether to apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(self._build_transforms(transforms))
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be set')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
            self.scale_key = 'scale'
            assert all(isinstance(scale, tuple) for scale in self.img_scale)
        else:
            self.img_scale = scale_factor if isinstance(scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'
        
        self.flip = flip
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert all(isinstance(direction, str) for direction in self.flip_direction)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn('flip_direction has no effect when flip is set to False')
        if self.flip and not any(t['type'] == 'RandomHorizontalFlip' for t in transforms):
            warnings.warn('flip has no effect when RandomHorizontalFlip is not in transforms')

    def _build_transforms(self, transform_list):
        """Build PaddlePaddle transforms from a list of dicts."""
        transform_dict = {
            'Resize': lambda x: Resize(size=x),
            'RandomHorizontalFlip': lambda: RandomHorizontalFlip(),
            'Normalize': lambda mean, std: Normalize(mean=mean, std=std),
            'Pad': lambda size_divisor: Pad(size_divisor=size_divisor),
            'ToTensor': lambda keys: ToTensor()
        }
        transforms = []
        for transform in transform_list:
            t_type = transform.pop('type')
            if t_type in transform_dict:
                transforms.append(transform_dict[t_type](**transform))
        return transforms

    def __call__(self, results):
        """Call function to apply test-time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction) for direction in self.flip_direction]
        
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
