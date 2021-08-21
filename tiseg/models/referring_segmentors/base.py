import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule

from ..builder import SEGMENTORS


def _parse_losses(losses):
    """Parse the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
            which may be a weighted sum of all losses, log_vars contains
            all the variables to be sent to the logger.
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


@SEGMENTORS.register_module()
class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def forward_train(self, img, txt, metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def forward_test(self, img, txt, metas, **kwargs):
        """Placeholder for Forward function for testing."""
        pass

    def forward(self, img, txt, metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, txt, metas, **kwargs)
        else:
            return self.forward_test(img, txt, metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        loss, log_vars = _parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch['metas']))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    def show_result(self,
                    img,
                    txt,
                    gt_instance_map,
                    pred_instance_map,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (np.ndarray): The image to be displayed.
            txt (str): The referring text to be displayed.
            gt_instance_map (np.ndarray): The ground truth instance map to be
                displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        seg_pred = pred_instance_map
        seg_label = gt_instance_map
        # TODO: PLAETTE support gradient color
        # Reference: `https://blog.csdn.net/qq_18351157/article/details/104093745` # noqa
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        color_seg_pred = img.copy()
        color_seg_label = img.copy()
        for label, color in enumerate(palette):
            color_seg_pred[seg_pred == label, :] = color
            color_seg_label[seg_label == label, :] = color[[0, 2, 1]]
        # convert to BGR
        color_seg_pred = color_seg_pred[..., ::-1]

        img_pred = img * (1 - opacity) + color_seg_pred * opacity
        img_label = img * (1 - opacity) + color_seg_label * opacity
        img_pred = img_pred.astype(np.uint8)
        img_label = img_label.astype(np.uint8)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        Hp, Wp, Cp = img_pred.shape
        Hl, Wl, Cl = img_label.shape
        assert Cp == Cl and Hp == Hl
        top_text_bar = np.zeros((50, Wp + Wl + 10, Cp)).astype(np.uint8)
        bottom_text_bar = np.zeros((50, Wp + Wl + 10, Cp)).astype(np.uint8)
        seg_line = np.zeros((Hp, 10, Cp)).astype(np.uint8)

        img = np.concatenate((img_pred, seg_line, img_label), axis=1)
        img = np.concatenate((top_text_bar, img, bottom_text_bar), axis=0)

        img = cv2.putText(img, 'Prediction', (10, 30),
                          cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 2, 255), 1)
        img = cv2.putText(img, 'GroundTruth', (10 + Wp, 30),
                          cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 2), 1)
        img = cv2.putText(img, txt, (10, 50 + Hp + 30),
                          cv2.FONT_HERSHEY_TRIPLEX, 0.5, (208, 216, 129), 1)

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show == False and out_file is not specified, only '
                          'result image will be returned')
            return img
