from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if img.dtype is not np.float32:
                img = img.astype(np.float32)
            results[key] = to_tensor(img.transpose(2, 0, 1))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class SegmapToTensor(object):
    """Convert segmentation map to :obj:`torch.Tensor` by given keys.

    The dimension order of input mask is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert segmentation map in results to
        :obj:`torch.Tensor` and transpose the channel order.

        Args:
            results (dict): Result dict contains the segmentation map data to
                convert.

        Returns:
            dict: The result dict contains the segmentation map converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            map = results[key]
            if len(map.shape) < 3:
                map = np.expand_dims(map, -1)
            if map.dtype is not np.uint8:
                map = map.astype(np.uint8)
            results[key] = to_tensor(map.transpose(2, 0, 1))

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Transpose(object):
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer(object):
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True),
            dict(key='gt_semantic_seg'))``.
    """

    def __init__(self,
                 fields=(dict(key='img',
                              stack=True), dict(key='gt_semantic_seg'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "gt_semantic_map". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_map: (1)unsqueeze dim(1), (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        img_raw = results['img']
        semantic_map = results['gt_semantic_map']
        semantic_map_with_edge = results['gt_semantic_map_with_edge']
        point_map = results['gt_point_map']
        direction_map = results['gt_direction_map']

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            if img.dtype is not np.float32:
                img = img.astype(np.float32)
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_map' in results:
            # convert to long
            results['gt_semantic_map'] = DC(
                to_tensor(results['gt_semantic_map'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        if 'gt_semantic_map_with_edge' in results:
            # convert to long
            results['gt_semantic_map_with_edge'] = DC(
                to_tensor(results['gt_semantic_map_with_edge'][None,
                                                               ...].astype(
                                                                   np.int64)),
                stack=True)
        if 'gt_point_map' in results:
            # convert to float
            results['gt_point_map'] = DC(
                to_tensor(results['gt_point_map'][None,
                                                  ...].astype(np.float32)),
                stack=True)
        if 'gt_direction_map' in results:
            # convert to long
            results['gt_direction_map'] = DC(
                to_tensor(results['gt_direction_map'][None,
                                                      ...].astype(np.int64)),
                stack=True)

        import matplotlib.pyplot as plt
        from tiseg.models.utils import generate_direction_differential_map
        direction_map_tensor = to_tensor(direction_map[None,
                                                       ...].astype(np.int64))
        direct_diff_map = generate_direction_differential_map(
            direction_map_tensor).numpy()[0]
        plt.figure(dpi=300)
        plt.subplot(231)
        plt.imshow(img_raw)
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(semantic_map)
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(semantic_map_with_edge)
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(point_map)
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(direction_map)
        plt.axis('off')
        plt.subplot(236)
        canvas = np.zeros((*semantic_map_with_edge.shape, 3), dtype=np.uint8)
        # canvas[direct_diff_map > 0, :] = (2, 255, 255)
        # canvas[semantic_map_with_edge == 1, :] = (255, 2, 255)
        canvas[semantic_map_with_edge > 0] = (2, 255, 255)
        canvas[semantic_map_with_edge == 9] = (0, 0, 255)
        canvas[direct_diff_map > 0] = (255, 0, 0)
        plt.imshow(canvas)
        plt.axis('off')
        plt.savefig('1.png')
        exit(0)

        return results

    def __repr__(self):
        return self.__class__.__name__


# TODO: Refactor doc string & comments
@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    Args:
    """

    def __init__(self,
                 data_keys,
                 label_keys,
                 meta_keys=('img_info', 'ann_info')):
        self.data_keys = data_keys
        self.label_keys = label_keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        meta = {}
        for key in self.meta_keys:
            meta[key] = results[key]
        data['metas'] = DC(meta, cpu_only=True)
        data['data'] = dict()
        data['label'] = dict()
        for key in self.data_keys:
            data['data'][key] = results[key]
        for key in self.label_keys:
            data['label'][key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
