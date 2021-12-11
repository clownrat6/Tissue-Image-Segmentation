import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import SEGMENTORS
from ..losses import MultiClassDiceLoss, miou, tiou
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor

from .try_cdnet_backbone import Unet


@SEGMENTORS.register_module()
class TryCDNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(TryCDNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = 8

        self.unet = Unet()

    def calculate(self, img):
        mask_logit, dir_logit, point_logit = self.unet(img)

        if self.test_cfg.get('use_ddm', False):
            # The whole image is too huge. So we use slide inference in
            # default.
            mask_logit = resize(input=mask_logit, size=self.test_cfg['plane_size'])
            direction_logit = resize(input=dir_logit, size=self.test_cfg['plane_size'])
            point_logit = resize(input=point_logit, size=self.test_cfg['plane_size'])

            mask_logit = self._ddm_enhencement(mask_logit, direction_logit, point_logit)
        mask_logit = resize(input=mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """

        if self.training:
            mask_logit, dir_logit, point_logit = self.unet(data['img'])
            assert label is not None
            mask_gt = label['sem_gt']
            point_gt = label['point_gt']
            dir_gt = label['dir_gt']

            loss = dict()
            mask_logit = resize(input=mask_logit, size=mask_gt.shape[2:])
            direction_logit = resize(input=dir_logit, size=dir_gt.shape[2:])
            point_logit = resize(input=point_logit, size=point_gt.shape[2:])

            mask_gt = mask_gt.squeeze(1)
            dir_gt = dir_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            mask_loss = self._mask_loss(mask_logit, mask_gt)
            loss.update(mask_loss)
            # direction branch loss calculation
            direction_loss = self._direction_loss(direction_logit, dir_gt)
            loss.update(direction_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, direction_logit, point_logit, mask_gt, dir_gt,
                                                         point_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            seg_logit = self.inference(data['img'], metas[0], True)
            seg_pred = seg_logit.argmax(dim=1)
            # Extract inside class
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred

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

        assert self.test_cfg.mode in ['slide', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        seg_logit_list = []

        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                rotate_num = (rotate_degree // 90) % 4
                img = torch.rot90(img, k=rotate_num, dims=(-2, -1))

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
                seg_logit = torch.rot90(seg_logit, k=rotate_num)

                seg_logit_list.append(seg_logit)

        seg_logit = sum(seg_logit_list) / len(seg_logit_list)

        return seg_logit

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

    def _mask_loss(self, mask_logit, mask_gt):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(mask_ce_loss_calculator(mask_logit, mask_gt))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_gt)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _direction_loss(self, direction_logit, dir_gt):
        direction_loss = {}
        direction_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        direction_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_angles + 1)
        direction_ce_loss = torch.mean(direction_ce_loss_calculator(direction_logit, dir_gt))
        direction_dice_loss = direction_dice_loss_calculator(direction_logit, dir_gt)
        # loss weight
        alpha = 1
        beta = 1
        direction_loss['direction_ce_loss'] = alpha * direction_ce_loss
        direction_loss['direction_dice_loss'] = beta * direction_dice_loss

        return direction_loss

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, mask_logit, direction_logit, point_logit, mask_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_gt = mask_gt.clone().detach()
        clean_direction_logit = direction_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['direction_miou'] = miou(clean_direction_logit, clean_dir_gt, self.num_angles + 1)

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['direction_tiou'] = tiou(clean_direction_logit, clean_dir_gt, self.num_angles + 1)

        # metric calculate
        mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        mask_pred[mask_pred == (self.num_classes - 1)] = 0
        mask_target = mask_gt.cpu().numpy().astype(np.uint8)
        mask_target[mask_target == (self.num_classes - 1)] = 0

        N = mask_pred.shape[0]
        wrap_dict['aji'] = 0.
        for i in range(N):
            aji_single_image = aggregated_jaccard_index(mask_pred[i], mask_target[i])
            wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # distributed environment requires cuda tensor
        wrap_dict['aji'] /= N
        wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict

    @classmethod
    def _ddm_enhencement(self, mask_logit, direction_logit, point_logit):
        # make direction differential map
        direction_map = torch.argmax(direction_logit, dim=1)
        direction_differential_map = generate_direction_differential_map(direction_map, 9)

        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        direction_differential_map[point_logit > 0.2] = 0

        # using direction differential map to enhance edge
        mask_logit = F.softmax(mask_logit, dim=1)
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] +
                                   direction_differential_map) * (1 + 2 * direction_differential_map)

        return mask_logit
