import torch
import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import MultiTaskUNetHead
from ..losses import MultiClassDiceLoss, BatchMultiClassDiceLoss, mdice, tdice
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class MultiTaskUNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(MultiTaskUNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = MultiTaskUNetHead(
            num_classes=[3, self.num_classes],
            mt_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        tc_mask_logit, mask_logit = self.head(bottom_feat, skip_feats)

        # tc_mask_logit: three-class mask logit
        # mask_logit: semantic mask logit
        return tc_mask_logit, mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            three_class_mask_logit, mask_logit = self.calculate(data['img'])
            img = data['img']
            downsampled_img = F.interpolate(img, mask_logit.shape[2:], mode='bilinear', align_corners=True)
            assert label is not None
            mask_label = label['sem_gt']
            tc_mask_label = label['sem_gt_w_bound']
            tc_mask_label[(tc_mask_label != 0) * (tc_mask_label != self.num_classes)] = 1
            tc_mask_label[tc_mask_label > 1] = 2
            loss = dict()
            mask_label = mask_label.squeeze(1)
            tc_mask_label = tc_mask_label.squeeze(1)
            mask_loss = self._mask_loss(downsampled_img, mask_logit, mask_label)
            loss.update(mask_loss)
            tc_mask_loss = self._tc_mask_loss(three_class_mask_logit, tc_mask_label)
            loss.update(tc_mask_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, mask_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            tc_seg_logit, seg_logit = self.inference(data['img'], metas[0], True)
            tc_seg_pred = tc_seg_logit.argmax(dim=1)
            seg_pred = seg_logit.argmax(dim=1)
            # Extract inside class
            tc_seg_pred = tc_seg_pred.cpu().numpy()
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            tc_seg_pred = list(tc_seg_pred)
            seg_pred = list(seg_pred)
            ret_list = []
            for tc_seg, seg in zip(tc_seg_pred, seg_pred):
                ret_list.append({'tc_sem_pred': tc_seg, 'sem_pred': seg})
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
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference patch or whole img
                if self.test_cfg.mode == 'split':
                    tc_sem_logit, sem_logit = self.split_inference(img, meta, rescale)
                else:
                    tc_sem_logit, sem_logit = self.whole_inference(img, meta, rescale)

                tc_sem_logit = self.reverse_tta_transform(tc_sem_logit, rotate_degree, flip_direction)
                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)

                tc_sem_logit = F.softmax(tc_sem_logit, dim=1)
                sem_logit = F.softmax(sem_logit, dim=1)

                tc_sem_logit_list.append(tc_sem_logit)
                sem_logit_list.append(sem_logit)

        tc_sem_logit = sum(tc_sem_logit_list) / len(tc_sem_logit_list)
        sem_logit = sum(sem_logit_list) / len(sem_logit_list)

        return tc_sem_logit, sem_logit

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
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                tc_patch, sem_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                tc_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = tc_patch[:, :, ind1_s - i:ind1_e - i,
                                                                        ind2_s - j:ind2_e - j]
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]

        tc_logit = tc_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        if rescale:
            tc_logit = resize(tc_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return tc_logit, sem_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        tc_logit, sem_logit = self.calculate(img)
        if rescale:
            tc_logit = resize(tc_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return tc_logit, sem_logit

    def _mask_loss(self, img, mask_logit, mask_label):
        """calculate semantic mask branch loss."""
        mask_loss = {}

        # loss weight
        alpha = 3
        beta = 1
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        mask_ce_loss = torch.mean(mask_ce_loss_calculator(mask_logit, mask_label))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_label)
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _tc_mask_loss(self, tc_mask_logit, tc_mask_label):
        """calculate three-class mask branch loss."""
        mask_loss = {}

        ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        dice_loss_calculator = MultiClassDiceLoss(num_classes=3)
        ce_loss = torch.mean(ce_loss_calculator(tc_mask_logit, tc_mask_label))
        dice_loss = dice_loss_calculator(tc_mask_logit, tc_mask_label)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['three_class_mask_ce_loss'] = alpha * ce_loss
        mask_loss['three_class_mask_dice_loss'] = beta * dice_loss

        return mask_loss

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()

        wrap_dict['mask_tdice'] = tdice(clean_mask_logit, clean_mask_label, self.num_classes)
        wrap_dict['mask_mdice'] = mdice(clean_mask_logit, clean_mask_label, self.num_classes)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # (the edge id is set `self.num_classes - 1` in default)
        # mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        # mask_pred[mask_pred == (self.num_classes - 1)] = 0
        # mask_target = mask_label.cpu().numpy().astype(np.uint8)
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
