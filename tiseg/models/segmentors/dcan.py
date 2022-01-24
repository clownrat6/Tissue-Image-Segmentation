import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..losses import mdice, tdice, BatchMultiClassDiceLoss
from .base import BaseSegmentor


def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


def up_convs(in_dims, out_dims, up_nums, norm_cfg=None, act_cfg=None):
    conv_list = []
    for idx in range(up_nums):
        if idx == 0:
            conv_list.extend([nn.Upsample(scale_factor=2), conv3x3(in_dims, out_dims, norm_cfg, act_cfg)])
        else:
            conv_list.extend([nn.Upsample(scale_factor=2), conv3x3(out_dims, out_dims, norm_cfg, act_cfg)])

    return nn.Sequential(*conv_list)


class BottleNeck(nn.Module):

    def __init__(self, in_dims):
        super().__init__()
        self.res_conv = nn.Sequential(*[
            conv1x1(in_dims // 4, in_dims // 4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            conv3x3(in_dims // 4, in_dims // 4, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            conv1x1(in_dims // 4, in_dims, norm_cfg=None, act_cfg=None),
        ])
        self.ide_conv = conv1x1(in_dims, in_dims, norm_cfg=dict(type='BN'), act_cfg=None)

        self.act = nn.ReLU()

    def forward(self, x):
        res = self.res_conv(x)
        ide = self.ide_conv(x)

        return self.act(res + ide)


@SEGMENTORS.register_module()
class DCAN(BaseSegmentor):
    """Implementation of `Deep Contour-Aware Networks`.
        Modified from https://github.com/chelovek21/BioImageSegmentation/blob/master/model.py
    """

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(DCAN, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])

        # NOTE: refine upsampling method.
        self.up_conv_4_cell = up_convs(512, num_classes - 1, 3)
        self.up_conv_4_cont = up_convs(512, 2, 3)

        self.up_conv_5_cell = up_convs(512, num_classes - 1, 4)
        self.up_conv_5_cont = up_convs(512, 2, 4)

        self.up_conv_6_cell = up_convs(512, num_classes - 1, 5)
        self.up_conv_6_cont = up_convs(512, 2, 5)

        # NOTE: raw upsampling method.
        # self.up_conv_4_cell = nn.Sequential(
        #     *[nn.ConvTranspose2d(b_dims * 8, num_classes - 1, kernel_size=16, stride=8)])
        # self.up_conv_4_cont = nn.Sequential(*[nn.ConvTranspose2d(b_dims * 8, 2, kernel_size=16, stride=8)])

        # self.up_conv_5_cell = nn.Sequential(
        #     *[nn.ConvTranspose2d(b_dims * 8, num_classes - 1, kernel_size=32, stride=16)])
        # self.up_conv_5_cont = nn.Sequential(*[nn.ConvTranspose2d(b_dims * 8, 2, kernel_size=32, stride=16)])

        # self.up_conv_6_cell = nn.Sequential(
        #     *[nn.ConvTranspose2d(b_dims * 16, num_classes - 1, kernel_size=64, stride=32)])
        # self.up_conv_6_cont = nn.Sequential(*[nn.ConvTranspose2d(b_dims * 16, 2, kernel_size=64, stride=32)])

    def calculate(self, img):
        img_feats = self.backbone(img)

        cell_4 = self.up_conv_4_cell(img_feats[-3])
        cont_4 = self.up_conv_4_cont(img_feats[-3])

        cell_5 = self.up_conv_5_cell(img_feats[-2])
        cont_5 = self.up_conv_5_cont(img_feats[-2])

        cell_6 = self.up_conv_6_cell(img_feats[-1])
        cont_6 = self.up_conv_6_cont(img_feats[-1])

        cell_logit = cell_4 + cell_5 + cell_6
        cont_logit = cont_4 + cont_5 + cont_6

        return cell_logit, cont_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            cell_logit, cont_logit = self.calculate(data['img'])
            assert label is not None
            mask_label = label['sem_gt']
            tc_mask_label = label['sem_gt_w_bound']
            tc_mask_label[(tc_mask_label != 0) * (tc_mask_label != (self.num_classes - 1))] = 1
            tc_mask_label[tc_mask_label > 1] = 2
            loss = dict()
            mask_label = mask_label.squeeze(1)
            tc_mask_label = tc_mask_label.squeeze(1)
            mask_loss = self._mask_loss(cell_logit, cont_logit, mask_label, tc_mask_label)
            loss.update(mask_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(cell_logit, mask_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            cell_logit, cont_logit = self.inference(data['img'], metas[0], True)
            cell_pred = cell_logit.argmax(dim=1)
            cont_pred = cont_logit.argmax(dim=1)
            seg_pred = cell_pred.clone()
            seg_pred[cont_pred > 0] = self.num_classes - 1
            # Extract inside class
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            ret_list = []
            for seg in seg_pred:
                ret_list.append({'sem_pred': seg})
            return ret_list

    def _mask_loss(self, cell_logit, cont_logit, mask_label, tc_mask_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes - 1)
        cont_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        cell_ce_loss = torch.mean(mask_ce_loss_calculator(cell_logit, mask_label.long()))
        cell_dice_loss = mask_dice_loss_calculator(cell_logit, mask_label.long())
        cont_ce_loss = torch.mean(mask_ce_loss_calculator(cont_logit, (tc_mask_label == 2).long()))
        cont_dice_loss = cont_dice_loss_calculator(cont_logit, (tc_mask_label == 2).long())
        # loss weight
        alpha = 5
        beta = 0.5
        mask_loss['cell_ce_loss'] = alpha * cell_ce_loss
        mask_loss['cont_ce_loss'] = alpha * cont_ce_loss
        mask_loss['cell_dice_loss'] = beta * cell_dice_loss
        mask_loss['cont_dice_loss'] = beta * cont_dice_loss

        return mask_loss

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
        cell_logit_list = []
        cont_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    cell_logit, cont_logit = self.split_inference(img, meta, rescale)
                else:
                    cell_logit, cont_logit = self.whole_inference(img, meta, rescale)

                cell_logit = self.reverse_tta_transform(cell_logit, rotate_degree, flip_direction)
                cont_logit = self.reverse_tta_transform(cont_logit, rotate_degree, flip_direction)

                cell_logit = F.softmax(cell_logit, dim=1)
                cont_logit = F.softmax(cont_logit, dim=1)

                cell_logit_list.append(cell_logit)
                cont_logit_list.append(cont_logit)

        cell_logit = sum(cell_logit_list) / len(cell_logit_list)
        cont_logit = sum(cont_logit_list) / len(cont_logit_list)

        if rescale:
            cell_logit = resize(cell_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            cont_logit = resize(cont_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return cell_logit, cont_logit

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

        cell_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        cont_logit = torch.zeros((B, 2, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                cell_patch, cont_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                cell_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = cell_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]
                cont_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = cont_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]

        cell_logit = cell_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        cont_logit = cont_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return cell_logit, cont_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        cell_logit, cont_logit = self.calculate(img)

        return cell_logit, cont_logit

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
