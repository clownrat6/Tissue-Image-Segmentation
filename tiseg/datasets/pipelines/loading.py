import imghdr
import json
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


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for referring expression segmentation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
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
        img_bytes = self.file_client.get(filename)
        gt_instance_map = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        # For referring expression segmentation, label map may be unnecessary
        # That's because map always has two classes - background and object.
        if results['ann_info'].get('label_map', None) is not None:
            for old_id, new_id in results['ann_info']['label_map'].items():
                gt_instance_map[gt_instance_map == old_id] = new_id
        results['gt_instance_map'] = gt_instance_map
        results['seg_fields'].append('gt_instance_map')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadTexts(object):
    """Load texts for referring expression segmentation.

    Convert sent string to text vector and support pad text vector.

    Args:
        pad_length (int, optional): The padding length of text vector. If this
            arg is not set, the length will remain unchanged.
            Default: None.
        pad_value (float, optional): The padding value of padding operation.
            This argument is only valid when pad_length is set.
            Default: 0.
    """

    def __init__(self, pad_length=None, pad_value=0):
        self.pad_length = pad_length
        self.pad_value = pad_value

    def __call__(self, results):
        """Call function to load sents and convert them to text vectors.

        Args:
            results (dict): Result dict from :obj:`tiseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded text vectors.
        """
        if results['sent_info'].get('sent_dir', None) is not None:
            filename = osp.join(results['sent_info']['sent_dir'],
                                results['sent_info']['sent_name'])
        else:
            filename = results['sent_info']['sent_name']

        sent_info = json.load(open(filename, 'r'))
        results['sent_info'].update(sent_info)
        # Encode sent str to text vector
        if self.pad_length is not None:
            txt_vector = self.pad_txt(results['sent_info']['txt'],
                                      self.pad_value, self.pad_length)
        else:
            txt_vector = results['sent_info']['txt']

        results['txt'] = txt_vector
        results['txt_info']['seq_len'] = self.pad_length
        results['txt_info']['pad_value'] = self.pad_value
        results['txt_info']['PAD_IDENTIFIER'] = '<pad>'
        return results

    @staticmethod
    def pad_txt(txt_vector, pad_value, pad_length):
        # Truncate long sentences
        if len(txt_vector) > pad_length:
            txt_vector = txt_vector[:pad_length]
        # Pad short sentences at the beginning with the special symbol '<pad>'
        # We set PAD_IDENTIFIER as 0 in default
        if len(txt_vector) < pad_length:
            txt_vector = [pad_value
                          ] * (pad_length - len(txt_vector)) + txt_vector
        return np.array(txt_vector)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pad_length={self.pad_length},'
        repr_str += f"pad_value='{self.pad_value}')"
        return repr_str
