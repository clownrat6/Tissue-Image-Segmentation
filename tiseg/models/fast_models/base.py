from abc import ABCMeta
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import BaseModule

from tiseg.utils import resize


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


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Segmentor supports multiplt data & multiple labels.

    For example (CDNet):
        Data: image;
        Label: semantic_map, semantic_map_with_edge, direction_map, point_map;

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

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

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch['metas']))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    # TODO refactor
    def slide_inference(self, img, meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.calculate(crop_img)
                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(preds, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return preds

    # TODO: refactor code stryle
    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = (self.test_cfg.crop_size[0] - self.test_cfg.stride[0]) * 2

        N, C, H, W = img.shape

        input = img

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
            tmp = torch.zeros((N, C, pad_h, W)).to(img.device)
            input = torch.cat((input, tmp), dim=2)

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
            tmp = torch.zeros((N, C, H + pad_h, pad_w)).to(img.device)
            input = torch.cat((input, tmp), dim=3)

        _, C1, H1, W1 = input.size()

        output = torch.zeros((input.size(0), 3, H1, W1)).to(img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch
                output_patch = self.calculate(input_var)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]

        output = output[:, :, :H, :W]
        if rescale:
            output = resize(output, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return output

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        seg_logit = self.calculate(img)
        if rescale:
            seg_logit = resize(seg_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return seg_logit

    def inference(self, img, meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict where each dict has: 'img_info',
                'ann_info'
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        raw_img = img
        assert self.test_cfg.mode in ['slide', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        seg_logit_list = []
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                rotate_num = (rotate_degree // 90) % 4
                img = torch.rot90(raw_img, k=rotate_num, dims=(-2, -1))

                if flip_direction == 'horizontal':
                    img = torch.flip(img, dims=[-1])
                if flip_direction == 'vertical':
                    img = torch.flip(img, dims=[-2])
                if flip_direction == 'diagonal':
                    img = torch.flip(img, dims=[-2, -1])

                if self.test_cfg.mode == 'slide':
                    seg_logit = self.slide_inference(img, meta, rescale)
                else:
                    seg_logit = self.whole_inference(img, meta, rescale)

                if flip_direction == 'horizontal':
                    seg_logit = torch.flip(seg_logit, dims=[-1])
                if flip_direction == 'vertical':
                    seg_logit = torch.flip(seg_logit, dims=[-2])
                if flip_direction == 'diagonal':
                    seg_logit = torch.flip(seg_logit, dims=[-2, -1])

                rotate_num = 4 - rotate_num
                seg_logit = torch.rot90(seg_logit, k=rotate_num, dims=(-2, -1))

                seg_logit_list.append(seg_logit)

        seg_logit = sum(seg_logit_list) / len(seg_logit_list)

        return seg_logit
