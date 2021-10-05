import imghdr
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_dir" and "img_info" (a dict that must contain the
    key "img_name"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`tiseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_info'].get('img_dir') is not None:
            filename = osp.join(results['img_info']['img_dir'],
                                results['img_info']['img_name'])
        else:
            filename = results['img_info']['img_name']

        if imghdr.what(filename) == 'gif':
            # The 19579.jpg image file of ImageClef has jpeg file suffix. But
            # the real file format is gif.
            import cv2
            cap = cv2.VideoCapture(filename)
            _, img = cap.read()
        else:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            if self.to_float32:
                img = img.astype(np.float32)

        results['img_info']['filename'] = filename
        results['img_info']['ori_filename'] = results['img_info']['img_name']
        results['img'] = img
        results['img_info']['img_shape'] = img.shape
        results['img_info']['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['img_info']['pad_shape'] = img.shape
        results['img_info']['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_info']['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


# TODO: Modify doc string & comments
@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load semantic level annotations.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 instance_suffix=None,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.instance_suffix = instance_suffix
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`tiseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded instance segmentation maps.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['ann_info'].get('ann_dir', None) is not None:
            filename = osp.join(results['ann_info']['ann_dir'],
                                results['ann_info']['ann_name'])
        else:
            filename = results['ann_info']['ann_name']
        suffix = osp.splitext(filename)[1]
        if suffix == '.npy':
            gt_semantic_map = np.load(filename)
        else:
            img_bytes = self.file_client.get(filename)
            gt_semantic_map = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)

        if self.instance_suffix is not None:
            extra_filename = filename.replace(
                results['ann_info']['ann_suffix'], self.instance_suffix)
            gt_instance_map = mmcv.imread(
                extra_filename,
                flag='unchanged',
                backend=self.imdecode_backend)
            results['gt_instance_map'] = gt_instance_map
            results['seg_fields'].append('gt_instance_map')
        results['gt_semantic_map'] = gt_semantic_map
        results['seg_fields'].append('gt_semantic_map')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(imdecode_backend='{self.imdecode_backend}')"
        return repr_str
