import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal',
                 rotate=False,
                 rotate_degree=90):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.img_ratios = img_ratios

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

        self.rotate = rotate
        self.rotate_degree = rotate_degree if isinstance(
            rotate_degree, list) else [rotate_degree]
        assert mmcv.is_list_of(self.rotate_degree, int)
        flag = any([t['type'] == 'RandomSparseRotate' for t in transforms])
        if (self.rotate and not flag):
            warnings.warn(
                'flip has no effect when RandomSparseRotate is not in '
                'transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        # Keep raw shape of gt_seg_map
        results['seg_fields'].clear()

        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            if h != w and self.rotate is True:
                warnings.warn(
                    'The image (height != width) is not suitable to do rotate '
                    'tta.')
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        # XXX: This code may return three flip direction which cause three
        # logit addition.
        flip_aug = [False, True] if self.flip else [False]
        self.flip_direction = self.flip_direction if self.flip else [
            'horizontal'
        ]
        rotate_aug = [False, True] if self.rotate else [False]
        self.rotate_degree = self.rotate_degree if self.rotate else [90]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    for rotate in rotate_aug:
                        for degree in self.rotate_degree:
                            # copy only can copy top-level object and can't
                            # copy son object.
                            _results = results.copy()
                            # only copy img & ann info (dict type)
                            _results['img_info'] = results['img_info'].copy()
                            _results['ann_info'] = results['ann_info'].copy()
                            _results['img_info']['scale'] = scale
                            _results['img_info']['flip'] = flip
                            _results['img_info']['flip_direction'] = direction
                            _results['img_info']['rotate'] = rotate
                            _results['img_info']['rotate_degree'] = degree
                            data = self.transforms(_results)
                            aug_data.append(data)

        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        # label is not supported now when evaluation
        aug_data_dict['label'] = []
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
