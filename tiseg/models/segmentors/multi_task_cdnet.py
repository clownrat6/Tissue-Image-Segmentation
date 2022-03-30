import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io, measure
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..heads.multi_task_cd_head import MultiTaskCDHead
from ..builder import SEGMENTORS
from ..losses import LossVariance, MultiClassBCELoss, BatchMultiClassSigmoidDiceLoss, MultiClassDiceLoss, TopologicalLoss, RobustFocalLoss2d, LevelsetLoss, ActiveContourLoss, mdice, tdice
from ..utils import generate_direction_differential_map, align_foreground
from .base import BaseSegmentor


def _convert_to_one_hot(tensor, bins, on_value=1, off_value=0):
    """Convert NxHxW shape tensor to NxCxHxW one-hot tensor.

    Args:
        tensor (torch.Tensor): The tensor to convert.
        bins (int): The number of one-hot channels.
            (`bins` is usually `num_classes + 1`)
        on_value (int): The one-hot activation value. Default: 1
        off_value (int): The one-hot deactivation value. Default: 0
    """
    assert tensor.ndim == 3
    assert on_value != off_value
    tensor_one_hot = F.one_hot(tensor, bins)
    tensor_one_hot[tensor_one_hot == 1] = on_value
    tensor_one_hot[tensor_one_hot == 0] = off_value

    return tensor_one_hot


class BatchMultiClassDiceLoss(nn.Module):
    """Calculate each class dice loss, then sum per class dice loss as a total
    loss."""

    def __init__(self, num_classes):
        super(BatchMultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logit, target, weights=None):
        assert target.ndim == 3
        # one-hot encoding for target
        target_one_hot = _convert_to_one_hot(target, self.num_classes).permute(0, 3, 1, 2).contiguous()
        smooth = 1e-4
        # softmax for logit
        logit = F.softmax(logit, dim=1)

        N, C, _, _ = target_one_hot.shape

        loss = 0
        ''' original         weights.shape = torch.Size([16, 1, 256, 256])
        for i in range(1, C):
            logit_per_class = logit[:, i]
            target_per_class = target_one_hot[:, i]

            intersection = logit_per_class * target_per_class
            # calculate per class dice loss
            dice_loss_per_class = (2 * intersection.sum((0, -2, -1)) + smooth) / (
                logit_per_class.sum((0, -2, -1)) + target_per_class.sum((0, -2, -1)) + smooth)
            # dice_loss_per_class = (2 * intersection.sum((0, -2, -1)) + smooth) / (
            #     logit_per_class.sum((0, -2, -1)) + target_per_class.sum((0, -2, -1)) + smooth)
            dice_loss_per_class = 1 - dice_loss_per_class
            if weights is not None:
                dice_loss_per_class *= weights[i]
            loss += dice_loss_per_class
        '''
        for i in range(1, C):
            if weights is not None:
                logit_per_class = logit[:, i]
                target_per_class = target_one_hot[:, i]
                # logit_per_class.shape=torch.Size([16, 256, 256]), target_per_class.shape=torch.Size([16, 256, 256])
                # weights.shape=torch.Size([16, 1, 256, 256])
                #print('logit_per_class.shape={}, target_per_class.shape={}'.format(logit_per_class.shape, target_per_class.shape))
                #print('intersection={}'.format(logit_per_class * target_per_class))
                #print('weights={}'.format(weights))
                #print('weights.shape={}'.format(weights.shape))
                intersection = (logit_per_class * target_per_class) * weights[:, 0]
                # calculate per class dice loss
                dice_loss_per_class = (2 * intersection.sum(
                    (0, -2, -1)) + smooth) / ((logit_per_class * weights[:, 0]).sum(
                        (0, -2, -1)) + (target_per_class * weights[:, 0]).sum((0, -2, -1)) + smooth)

            else:
                logit_per_class = logit[:, i]
                target_per_class = target_one_hot[:, i]

                intersection = (logit_per_class * target_per_class)
                # calculate per class dice loss

                dice_loss_per_class = (2 * intersection.sum((0, -2, -1)) + smooth) / (
                    logit_per_class.sum((0, -2, -1)) + target_per_class.sum((0, -2, -1)) + smooth)
            #print('intersection.sum={}'.format(intersection.sum((0, -2, -1))))
            #print('logit_per_class.sum={}'.format(logit_per_class.sum((0, -2, -1))))
            #print('target_per_class.sum={}'.format(target_per_class.sum((0, -2, -1))))
            #print('dice_loss_per_class={}'.format(dice_loss_per_class))
            #a = dice_loss_per_class[0,0,0,] # 让他报错
            dice_loss_per_class = 1 - dice_loss_per_class

            loss += dice_loss_per_class

        return loss


@SEGMENTORS.register_module()
class MultiTaskCDNet(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(MultiTaskCDNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = self.train_cfg.get('num_angles', 8)

        # argument
        self.if_ddm = self.test_cfg.get('if_ddm', False)
        self.use_modify_dirloss = self.train_cfg.get('use_modify_dirloss', False)
        self.use_ac = self.train_cfg.get('use_ac', False)
        self.use_sigmoid = self.train_cfg.get('use_sigmoid', False)
        self.use_tploss = self.train_cfg.get('use_tploss', False)
        self.tploss_weight = self.train_cfg.get('tploss_weight', False)
        self.tploss_dice = self.train_cfg.get('tploss_dice', False)
        self.use_ac = self.train_cfg.get('use_ac', False)
        self.ac_len_weight = self.train_cfg.get('ac_len_weight', 1)
        self.use_variance = self.train_cfg.get('use_variance', True)
        self.use_focal = self.train_cfg.get('use_focal', False)
        self.use_level = self.train_cfg.get('use_level', False)

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = MultiTaskCDHead(
            num_classes=self.num_classes,
            num_angles=8,
            dgm_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            noau=True,
            use_regression=False,
            parallel=True)

    def calculate(self, img, rescale=False):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        tc_logit, sem_logit, dir_logit, point_logit = self.head(bottom_feat, skip_feats)

        return tc_logit, sem_logit, dir_logit, point_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            tc_logit, sem_logit, dir_logit, point_logit = self.calculate(data['img'])
            img = data['img']
            downsampled_img = F.interpolate(img, sem_logit.shape[2:], mode='bilinear', align_corners=True)
            assert label is not None
            sem_gt = label['sem_gt']
            sem_gt_wb = label['sem_gt_w_bound']
            tc_gt = sem_gt_wb.clone()
            tc_gt[(tc_gt != 0) * (tc_gt != self.num_classes)] = 1
            tc_gt[tc_gt > 1] = 2
            inst_gt = label['inst_gt']
            point_gt = label['point_gt']
            dir_gt = label['dir_gt']
            weight_map = label['loss_weight_map'] if self.use_modify_dirloss else None

            loss = dict()

            tc_gt = tc_gt.squeeze(1)
            sem_gt = sem_gt.squeeze(1)
            dir_gt = dir_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            sem_loss = self._sem_loss(downsampled_img, sem_logit, sem_gt, inst_gt)
            loss.update(sem_loss)
            # three classes mask branch loss calculation
            tc_loss = self._tc_loss(tc_logit, tc_gt)
            # tc_loss = self._tc_loss(tc_logit, tc_gt, weight_map)
            loss.update(tc_loss)
            # direction branch loss calculation
            dir_loss = self._dir_loss(dir_logit, dir_gt, tc_logit, tc_gt, weight_map)
            loss.update(dir_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, dir_logit, point_logit, sem_gt, dir_gt, point_gt)
            loss.update(training_metric_dict)

            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            tc_logit, sem_logit, dir_map = self.inference(data['img'], metas[0], True)
            tc_pred = tc_logit.argmax(dim=1)
            sem_pred = sem_logit.argmax(dim=1)
            tc_pred = tc_pred.to('cpu').numpy()[0]
            sem_pred = sem_pred.to('cpu').numpy()[0]
            sem_pred, inst_pred = self.postprocess(tc_pred, sem_pred)
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, tc_pred, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(sem_pred))
        sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = sem_pred == sem_id
            # fill instance holes
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_canvas[sem_id_mask > 0] = sem_id

        # instance process & dilation
        bin_pred = tc_pred.copy()
        bin_pred[bin_pred == 2] = 0

        inst_pred = measure.label(bin_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        # inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))
        inst_pred = align_foreground(inst_pred, sem_canvas > 0, 20)

        return sem_pred, inst_pred

    def inference(self, img, meta, rescale):
        """Inference with split/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        tc_logit_list = []
        sem_logit_list = []
        dir_logit_list = []
        point_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    tc_logit, sem_logit, dir_logit, point_logit = self.split_inference(img, meta, rescale)
                else:
                    tc_logit, sem_logit, dir_logit, point_logit = self.whole_inference(img, meta, rescale)

                tc_logit = self.reverse_tta_transform(tc_logit, rotate_degree, flip_direction)
                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dir_logit = self.reverse_tta_transform(dir_logit, rotate_degree, flip_direction)
                point_logit = self.reverse_tta_transform(point_logit, rotate_degree, flip_direction)

                tc_logit = F.softmax(tc_logit, dim=1)
                sem_logit = F.softmax(sem_logit, dim=1)
                dir_logit = F.softmax(dir_logit, dim=1)

                tc_logit_list.append(tc_logit)
                sem_logit_list.append(sem_logit)
                dir_logit_list.append(dir_logit)
                point_logit_list.append(point_logit)

        tc_logit = sum(tc_logit_list) / len(tc_logit_list)
        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        point_logit = sum(point_logit_list) / len(point_logit_list)

        if rescale:
            tc_logit = resize(tc_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        dd_map_list = []
        dir_map_list = []
        for dir_logit in dir_logit_list:
            if rescale:
                dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_logit[:, 0] = dir_logit[:, 0] * tc_logit[:, 0]
            dir_map = torch.argmax(dir_logit, dim=1)
            dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)
            dir_map_list.append(dir_map)
            dd_map_list.append(dd_map)

        dd_map = sum(dd_map_list) / len(dd_map_list)

        if self.if_ddm:
            tc_logit = self._ddm_enhencement(tc_logit, dd_map, point_logit)

        return tc_logit, sem_logit, dir_map_list[0]

    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
        else:
            pad_h = window_size - H

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
        else:
            pad_w = window_size - W

        H1, W1 = pad_h + H, pad_w + W
        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        tc_logit = torch.zeros((B, 3, H1, W1), dtype=img.dtype, device=img.device)
        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        dir_logit = torch.zeros((B, self.num_angles + 1, H1, W1), dtype=img.dtype, device=img.device)
        point_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                tc_sem_patch, sem_patch, dir_patch, point_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                tc_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = tc_sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                dir_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = dir_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                point_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = point_patch[:, :, ind1_s - i:ind1_e - i,
                                                                              ind2_s - j:ind2_e - j]

        tc_logit = tc_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        dir_logit = dir_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        point_logit = point_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return tc_logit, sem_logit, dir_logit, point_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        tc_logit, sem_logit, dir_logit, point_logit = self.calculate(img)

        return tc_logit, sem_logit, dir_logit, point_logit

    def _tc_loss(self, tc_logit, tc_gt, weight_map=None):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = MultiClassDiceLoss(num_classes=3)
        # Assign weight map for each pixel position
        mask_ce_loss = mask_ce_loss_calculator(tc_logit, tc_gt)
        if weight_map is not None:
            mask_ce_loss *= weight_map[:, 0]
        mask_ce_loss = torch.mean(mask_ce_loss)
        mask_dice_loss = mask_dice_loss_calculator(tc_logit, tc_gt)
        # loss weight
        alpha = 3
        beta = 1
        mask_loss['tc_ce_loss'] = alpha * mask_ce_loss
        mask_loss['tc_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _sem_loss(self, img, sem_logit, sem_gt, inst_gt=None):
        """calculate semantic mask branch loss."""
        mask_loss = {}

        # loss weight
        alpha = 3
        beta = 1
        gamma = 5
        assert not (self.use_focal and self.use_level
                    and self.use_ac), 'Can\'t use focal loss & deep level set loss at the same time.'
        if self.use_sigmoid:
            if self.use_ac:
                ac_w_area = self.train_cfg.get('ac_w_area')
                ac_loss_calculator = ActiveContourLoss(w_area=ac_w_area, len_weight=self.ac_len_weight)
                ac_loss_collect = []
                for i in range(1, self.num_classes):
                    sem_logit_cls = sem_logit[:, i:i + 1].sigmoid()
                    sem_gt_cls = (sem_gt == i)[:, None].float()
                    ac_loss_collect.append(ac_loss_calculator(sem_logit_cls, sem_gt_cls))
                mask_loss['mask_ac_loss'] = gamma * (sum(ac_loss_collect) / len(ac_loss_collect))
            else:
                mask_bce_loss_calculator = MultiClassBCELoss(num_classes=self.num_classes)
                mask_dice_loss_calculator = BatchMultiClassSigmoidDiceLoss(num_classes=self.num_classes)
                mask_bce_loss = mask_bce_loss_calculator(sem_logit, sem_gt)
                mask_dice_loss = mask_dice_loss_calculator(sem_logit, sem_gt)
                mask_loss['mask_bce_loss'] = alpha * mask_bce_loss
                mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        else:
            if self.use_focal:
                mask_focal_loss_calculator = RobustFocalLoss2d(type='softmax')
                mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
                mask_focal_loss = mask_focal_loss_calculator(sem_logit, sem_gt)
                mask_dice_loss = mask_dice_loss_calculator(sem_logit, sem_gt)
                mask_loss['mask_focal_loss'] = alpha * mask_focal_loss
                mask_loss['mask_dice_loss'] = beta * mask_dice_loss
            else:
                mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
                mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
                mask_ce_loss = torch.mean(mask_ce_loss_calculator(sem_logit, sem_gt))
                mask_dice_loss = mask_dice_loss_calculator(sem_logit, sem_gt)
                mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
                mask_loss['mask_dice_loss'] = beta * mask_dice_loss

            sem_logit = sem_logit.softmax(dim=1)
            if self.use_ac:
                ac_w_area = self.train_cfg.get('ac_w_area')
                ac_loss_calculator = ActiveContourLoss(w_area=ac_w_area, len_weight=self.ac_len_weight)
                ac_loss_collect = []
                for i in range(1, self.num_classes):
                    sem_logit_cls = sem_logit[:, i:i + 1]
                    sem_gt_cls = (sem_gt == i)[:, None].float()
                    ac_loss_collect.append(ac_loss_calculator(sem_logit_cls, sem_gt_cls))
                mask_loss['mask_ac_loss'] = 4 * gamma * sum(ac_loss_collect) / len(ac_loss_collect)
            if self.use_variance:
                vvv = gamma / 3
                variance_loss_calculator = LossVariance()
                variance_loss = variance_loss_calculator(sem_logit, inst_gt[:, 0])
                mask_loss['mask_variance_loss'] = vvv * variance_loss

        if self.use_level:
            # calculate deep level set loss for each semantic class.
            loss_collect = []
            weights = [1 for i in range(1, self.num_classes)]
            for i in range(1, self.num_classes):
                sem_logit_cls = sem_logit[:, i:i + 1].sigmoid()
                bg_sem_logit_cls = -sem_logit[:, i:i + 1].sigmoid()
                overall_sem_logits = torch.cat([sem_logit_cls, bg_sem_logit_cls], dim=1)
                sem_gt_cls = (sem_gt == i)[:, None]
                overall_sem_logits = overall_sem_logits * sem_gt_cls
                img_region = sem_gt_cls * img
                level_loss_calculator = LevelsetLoss()
                loss_collect.append(level_loss_calculator(sem_logit_cls, img_region, weights[i]))
            mask_loss['mask_level_loss'] = sum(loss_collect) / len(loss_collect)

        return mask_loss

    def _dir_loss(self, dir_logit, dir_gt, tc_logit=None, tc_gt=None, weight_map=None):
        dir_loss = {}
        if self.use_modify_dirloss:
            # weight_map.shape=torch.Size([16, 1, 256, 256])
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            # dir_dice_loss_calculator = WeightMulticlassDiceLoss(num_classes=self.num_angles + 1)
            dir_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_angles + 1)
            # Assign weight map for each pixel position
            dir_ce_loss = dir_ce_loss_calculator(dir_logit, dir_gt)
            dir_ce_loss *= weight_map[:, 0]
            dir_ce_loss = torch.mean(dir_ce_loss)
            dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt, weight_map)
            # dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt)
            # loss weight
            alpha = 1
            beta = 1
            dir_loss['dir_ce_loss'] = alpha * dir_ce_loss
            dir_loss['dir_dice_loss'] = beta * dir_dice_loss
        else:
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            dir_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_angles + 1)
            # Assign weight map for each pixel position
            dir_ce_loss = dir_ce_loss_calculator(dir_logit, dir_gt)
            if weight_map is not None:
                dir_ce_loss *= weight_map[:, 0]
            dir_ce_loss = torch.mean(dir_ce_loss)
            dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt)
            # loss weight
            alpha = 3
            beta = 1
            dir_loss['dir_ce_loss'] = alpha * dir_ce_loss
            dir_loss['dir_dice_loss'] = beta * dir_dice_loss

        # use_tploss = self.train_cfg.get('use_tploss', False)
        # tploss_weight = self.train_cfg.get('tploss_weight', False)
        # tploss_dice = self.train_cfg.get('tploss_dice', False)
        if self.use_tploss:
            pred_contour = torch.argmax(tc_logit, dim=1) == 2  # [B, H, W]
            gt_contour = tc_gt == 2
            dir_tp_loss_calculator = TopologicalLoss(
                use_regression=False, weight=self.tploss_weight, num_angles=self.num_angles, use_dice=self.tploss_dice)
            dir_tp_loss = dir_tp_loss_calculator(dir_logit, dir_gt, pred_contour, gt_contour)
            theta = 1
            dir_loss['dir_tp_loss'] = dir_tp_loss * theta

        return dir_loss

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        # alpha = 5 # MoNuSeg
        alpha = 3  # 1  # CoNIC
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, sem_logit, dir_logit, point_logit, sem_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_gt = sem_gt.clone().detach()

        wrap_dict['mask_tdice'] = tdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['mask_mdice'] = mdice(clean_sem_logit, clean_sem_gt, self.num_classes)

        clean_dir_logit = dir_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()
        wrap_dict['dir_tdice'] = tdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)
        wrap_dict['dir_mdice'] = mdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # mask_pred = torch.argmax(sem_logit, dim=1).cpu().numpy().astype(np.uint8)
        # mask_pred[mask_pred == (self.num_classes - 1)] = 0
        # mask_target = sem_gt.cpu().numpy().astype(np.uint8)
        # mask_target[mask_target == (self.num_classes - 1)] = 0

        # N = mask_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(mask_pred[i], mask_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict

    @classmethod
    def _ddm_enhencement(self, sem_logit, dd_map, point_logit):
        stamp = time.time()  # 获取当前时间戳
        stamp = str(stamp)[:10]
        # using point map to remove center point direction differential map
        print('sem_logit.shape={}, point_logit.shape={}, point_logit[:, 0, :, :].shape={}'.format(
            sem_logit.shape, point_logit.shape, point_logit[:, 0, :, :].shape))
        print('point_logit[:, 0, :, :].unique={}'.format(np.unique(point_logit[:, 0, :, :].detach().cpu().numpy())))

        point_logit = point_logit[:, 0, :, :]
        dist_map = point_logit + 0.2
        foreground_prob = (dist_map / torch.max(dist_map))**2
        foreground_map = foreground_prob > 0.6  # 0.2  0.7
        # point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        weight_map0 = (1 - foreground_prob)
        weight_map1 = dd_map * weight_map0
        weight_map2 = dd_map * (1 - (dist_map / 10)**(5))
        # mask out some redundant direction differential
        dd_map1 = dd_map - (dd_map * foreground_map)
        dd_map2 = dd_map * weight_map1
        ''''''
        print('weight_map0.unique={}'.format(np.unique(weight_map0[0, :, :].detach().cpu().numpy())))
        print('weight_map1.unique={}'.format(np.unique(weight_map1[0, :, :].detach().cpu().numpy())))
        print('weight_map2.unique={}'.format(np.unique(weight_map2[0, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_weight_map0.png'.format(str(stamp)), weight_map0[0, :, :].detach().cpu().numpy() * 50)
        io.imsave('./work_vis/{}_weight_map1.png'.format(str(stamp)), weight_map1[0, :, :].detach().cpu().numpy() * 50)
        io.imsave('./work_vis/{}_weight_map2.png'.format(str(stamp)), weight_map2[0, :, :].detach().cpu().numpy() * 50)

        io.imsave('./work_vis/{}_point_logit0.png'.format(str(stamp)), point_logit[0, :, :].detach().cpu().numpy() * 25)
        io.imsave('./work_vis/{}_point_logit0_foreground_map.png'.format(str(stamp)),
                  foreground_map[0, :, :].detach().cpu().numpy() * 200)
        print('foreground_map.unique={}'.format(np.unique(foreground_map[0, :, :].detach().cpu().numpy())))

        print('dd_map0.unique={}'.format(np.unique(dd_map[0, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_dd_map0.png'.format(str(stamp)), dd_map[0, :, :].detach().cpu().numpy() * 255)

        print('dd_map1.unique={}'.format(np.unique(dd_map1[0, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_dd_map1.png'.format(str(stamp)), dd_map1[0, :, :].detach().cpu().numpy() * 255)

        print('dd_map2.unique={}'.format(np.unique(dd_map2[0, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_dd_map2.png'.format(str(stamp)), dd_map2[0, :, :].detach().cpu().numpy() * 255)

        print('sem_logit00.unique={}'.format(np.unique(sem_logit[:, -1, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_sem_logit_0.png'.format(str(stamp)),
                  sem_logit[0, -1, :, :].detach().cpu().numpy() * 50)
        print('sem_logit000.unique={}'.format(np.unique(sem_logit[:, -2, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_sem_logit_maskprob.png'.format(str(stamp)),
                  sem_logit[0, -2, :, :].detach().cpu().numpy() * 50)

        # using direction differential map to enhance edge
        #sem_logit[:, -1, :, :] = (sem_logit[:, -1, :, :] + dd_map1) * (1 + dd_map1)
        sem_logit[:, -1, :, :] = (sem_logit[:, -1, :, :]) * (1 + (dd_map1)) * weight_map0
        #sem_logit[:, -1, :, :] = (sem_logit[:, -1, :, :] + dd_map1/3) * (1 + (dd_map1+dd_map2)/3)

        sem_logit[:, -1, :, :][sem_logit[:, -1, :, :] >= 1] = 0.95
        sem_logit[:, -2, :, :][foreground_map == 0.8] = 1

        print('sem_logit01.unique={}'.format(np.unique(sem_logit[:, -1, :, :].detach().cpu().numpy())))
        io.imsave('./work_vis/{}_sem_logit_1.png'.format(str(stamp)),
                  sem_logit[0, -1, :, :].detach().cpu().numpy() * 50)
        #a = sem_logit[0,0,0,0,0] # 报错，停止符
        return sem_logit
