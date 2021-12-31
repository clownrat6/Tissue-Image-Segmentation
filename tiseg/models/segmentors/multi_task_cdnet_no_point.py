import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..heads import MultiTaskCDHeadNoPoint
from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss, MultiClassDiceLoss, mdice, tdice
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor
from ...datasets.utils import (angle_to_vector, vector_to_label)


@SEGMENTORS.register_module()
class MultiTaskCDNetSegmentorNoPoint(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(MultiTaskCDNetSegmentorNoPoint, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        

        # argument
        self.if_weighted_loss = self.train_cfg.get('if_weighted_loss', False)
        self.if_ddm = self.test_cfg.get('if_ddm', False)
        self.if_mudslide = self.test_cfg.get('if_mudslide', False)
        self.num_angles = self.train_cfg.get('num_angles', 8)
        self.use_regression = self.train_cfg.get('use_regression', False)
        self.use_semantic = self.train_cfg.get('use_semantic', True)

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = MultiTaskCDHeadNoPoint(
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            dgm_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            noau=self.train_cfg.get('noau', False),
            use_regression=self.use_regression)

    def calculate(self, img, rescale=False):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        tc_mask_logit, mask_logit, dir_logit = self.head(bottom_feat, skip_feats)

        if rescale:
            tc_mask_logit = resize(input=tc_mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
            mask_logit = resize(input=mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
            dir_logit = resize(input=dir_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return tc_mask_logit, mask_logit, dir_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            tc_mask_logit, mask_logit, dir_logit = self.calculate(data['img'])

            assert label is not None
            tc_mask_gt = label['sem_gt_w_bound']
            tc_mask_gt[(tc_mask_gt != 0) * (tc_mask_gt != self.num_classes)] = 1
            tc_mask_gt[tc_mask_gt > 1] = 2
            mask_gt = label['sem_gt']
            if self.use_regression:
                dir_gt = label['reg_dir_gt']
            else:
                dir_gt = label['dir_gt']
                dir_gt = dir_gt.squeeze(1)
            weight_map = label['loss_weight_map'] if self.if_weighted_loss else None

            loss = dict()

            tc_mask_gt = tc_mask_gt.squeeze(1)
            mask_gt = mask_gt.squeeze(1)
            

            # TODO: Conside to remove some edge loss value.

            # mask branch loss calculation
            if self.use_semantic:
                mask_loss = self._mask_loss(mask_logit, mask_gt, None)
                loss.update(mask_loss)

            # three classes mask branch loss calculation
            tc_mask_loss = self._tc_mask_loss(tc_mask_logit, tc_mask_gt, weight_map)
            loss.update(tc_mask_loss)
            # direction branch loss calculation
            dir_loss = self._dir_loss(dir_logit, dir_gt, weight_map)
            loss.update(dir_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, dir_logit, tc_mask_logit, mask_gt, dir_gt, tc_mask_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            tc_seg_logit, seg_logit, dir_map = self.inference(data['img'], metas[0], True)
            tc_seg_pred = tc_seg_logit.argmax(dim=1)
            seg_pred = seg_logit.argmax(dim=1)
            tc_seg_pred = tc_seg_pred.to('cpu').numpy()
            seg_pred = seg_pred.to('cpu').numpy()
            dir_map = dir_map.to('cpu').numpy()
            # unravel batch dim
            tc_seg_pred = list(tc_seg_pred)
            seg_pred = list(seg_pred)
            dir_map = list(dir_map)
            ret_list = []
            for tc_seg, seg, dir in zip(tc_seg_pred, seg_pred, dir_map):
                ret_dict = {'tc_sem_pred': tc_seg}
                if self.use_semantic:
                    ret_dict['sem_pred'] = seg
                if self.if_mudslide:
                    ret_dict['dir_pred'] = dir
                ret_list.append(ret_dict)
            return ret_list

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
        tc_sem_logit_list = []
        sem_logit_list = []
        dir_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    tc_sem_logit, sem_logit, dir_logit = self.split_inference(img, meta, rescale)
                else:
                    tc_sem_logit, sem_logit, dir_logit = self.whole_inference(img, meta, rescale)

                tc_sem_logit = self.reverse_tta_transform(tc_sem_logit, rotate_degree, flip_direction)
                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dir_logit = self.reverse_tta_transform(dir_logit, rotate_degree, flip_direction)

                tc_sem_logit = F.softmax(tc_sem_logit, dim=1)
                sem_logit = F.softmax(sem_logit, dim=1)
                if not self.use_regression:
                    dir_logit = F.softmax(dir_logit, dim=1)

                tc_sem_logit_list.append(tc_sem_logit)
                sem_logit_list.append(sem_logit)
                dir_logit_list.append(dir_logit)

        tc_sem_logit = sum(tc_sem_logit_list) / len(tc_sem_logit_list)
        sem_logit = sum(sem_logit_list) / len(sem_logit_list)

        dd_map_list = []
        dir_map_list = []
        for dir_logit in dir_logit_list:
            if self.use_regression:
                dir_logit[dir_logit < 0] = 0
                dir_logit[dir_logit > 2 * np.pi] = 2 * np.pi
                background = (torch.argmax(tc_sem_logit, dim=1)[0] == 0).cpu().numpy()
                angle_map = dir_logit * 180 / np.pi
                angle_map = angle_map[0, 0].cpu().numpy()  #[H, W]
                angle_map[angle_map > 180] -= 360
                angle_map[background] = 0
                vector_map = angle_to_vector(angle_map, self.num_angles)
                dir_map = vector_to_label(vector_map, self.num_angles)
                dir_map[background] = -1
                dir_map = dir_map + 1
                dir_map = torch.from_numpy(dir_map[None, :, :]).cuda()
                dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)
            else:
                dir_logit[:, 0] = dir_logit[:, 0] * tc_sem_logit[:, 0]
                dir_map = torch.argmax(dir_logit, dim=1)
                if self.num_angles == 8:
                    dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)
                else:
                    dd_map = torch.zeros_like(dir_map).cuda()
            dir_map_list.append(dir_map)
            dd_map_list.append(dd_map)

        dd_map = sum(dd_map_list) / len(dd_map_list)

        if self.if_ddm:
            tc_sem_logit = self._ddm_enhencement(tc_sem_logit, dd_map)

        return tc_sem_logit, sem_logit, dir_map_list[0]

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

        tc_sem_logit = torch.zeros((B, 3, H1, W1), dtype=img.dtype, device=img.device)
        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        if self.use_regression:
            dir_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        else:
            dir_logit = torch.zeros((B, self.num_angles + 1, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                tc_sem_patch, sem_patch, dir_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                tc_sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = tc_sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                                ind2_s - j:ind2_e - j]
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                dir_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = dir_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]

        tc_sem_logit = tc_sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        dir_logit = dir_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        if rescale:
            tc_sem_logit = resize(tc_sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return tc_sem_logit, sem_logit, dir_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        tc_sem_logit, sem_logit, dir_logit = self.calculate(img)
        if rescale:
            tc_sem_logit = resize(tc_sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return tc_sem_logit, sem_logit, dir_logit

    def _tc_mask_loss(self, tc_mask_logit, tc_mask_gt, weight_map=None):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = MultiClassDiceLoss(num_classes=3)
        # Assign weight map for each pixel position
        mask_ce_loss = mask_ce_loss_calculator(tc_mask_logit, tc_mask_gt)
        if weight_map is not None:
            mask_ce_loss *= weight_map[:, 0]
        mask_ce_loss = torch.mean(mask_ce_loss)
        mask_dice_loss = mask_dice_loss_calculator(tc_mask_logit, tc_mask_gt)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['tc_mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['tc_mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _mask_loss(self, mask_logit, mask_gt, weight_map=None):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        mask_ce_loss = mask_ce_loss_calculator(mask_logit, mask_gt)
        if weight_map is not None:
            mask_ce_loss *= weight_map[:, 0]
        mask_ce_loss = torch.mean(mask_ce_loss)
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_gt)
        # loss weight
        alpha = 3
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _dir_loss(self, dir_logit, dir_gt, weight_map=None):
        dir_loss = {}
        if self.use_regression:
            dir_mse_loss_calculator = nn.MSELoss(reduction='none')
            dir_degree_mse_loss = dir_mse_loss_calculator(dir_logit, dir_gt)
            if weight_map is not None:
                dir_degree_mse_loss *= weight_map[:, 0]
            dir_degree_mse_loss = torch.mean(dir_degree_mse_loss)
            dir_loss['dir_degree_mse_loss'] = dir_degree_mse_loss
        else:
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            dir_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_angles + 1)
            # Assign weight map for each pixel position
            dir_ce_loss = dir_ce_loss_calculator(dir_logit, dir_gt)
            if weight_map is not None:
                dir_ce_loss *= weight_map[:, 0]
            dir_ce_loss = torch.mean(dir_ce_loss)
            dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt)
            # loss weight
            alpha = 1
            beta = 1
            dir_loss['dir_ce_loss'] = alpha * dir_ce_loss
            dir_loss['dir_dice_loss'] = beta * dir_dice_loss

        return dir_loss

    def _training_metric(self, mask_logit, dir_logit, tc_mask_logit, mask_gt, dir_gt, tc_mask_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        if self.use_semantic:
            clean_mask_logit = mask_logit.clone().detach()
            clean_mask_gt = mask_gt.clone().detach()
            wrap_dict['mask_mdice'] = mdice(clean_mask_logit, clean_mask_gt, self.num_classes)
            wrap_dict['mask_tdice'] = tdice(clean_mask_logit, clean_mask_gt, self.num_classes)
        else:
            clean_mask_logit = tc_mask_logit.clone().detach()
            clean_mask_gt = tc_mask_gt.clone().detach()
            wrap_dict['tc_mask_mdice'] = mdice(clean_mask_logit, clean_mask_gt, 3)
            wrap_dict['tc_mask_tdice'] = tdice(clean_mask_logit, clean_mask_gt, 3)

        if not self.use_regression:
            clean_dir_logit = dir_logit.clone().detach()
            clean_dir_gt = dir_gt.clone().detach()
            wrap_dict['dir_mdice'] = mdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)
            wrap_dict['dir_tdice'] = tdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        # mask_pred[mask_pred == (self.num_classes - 1)] = 0
        # mask_target = mask_gt.cpu().numpy().astype(np.uint8)
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
    def _ddm_enhencement(self, mask_logit, dd_map):
        # using direction differential map to enhance edge
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] + dd_map) * (1 + dd_map)

        return mask_logit
