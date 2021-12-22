import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from skimage import io

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..heads.regression_degree_cd_head import RegDegreeCDHead
from ..builder import SEGMENTORS
from ..losses import MultiClassDiceLoss, miou, tiou
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor
from ...datasets.utils import (angle_to_vector, calculate_centerpoint, calculate_gradient, vector_to_label)


@SEGMENTORS.register_module()
class RegDegreeCDNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(RegDegreeCDNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = 8

        # argument
        self.if_weighted_loss = self.train_cfg.get('if_weighted_loss', False)
        self.if_ddm = self.test_cfg.get('if_ddm', False)
        self.if_mudslide = self.test_cfg.get('if_mudslide', False)

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = RegDegreeCDHead(
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            dgm_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img, rescale=False):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        mask_logit, dir_degree_logit, point_logit = self.head(bottom_feat, skip_feats)

        if rescale:
            mask_logit = resize(input=mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
            dir_degree_logit = resize(input=dir_degree_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
            point_logit = resize(input=point_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return mask_logit, dir_degree_logit, point_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        # print(data)
        # print(metas)
        if self.training:
            mask_logit, dir_degree_logit, point_logit = self.calculate(data['img'])

            assert label is not None
            mask_gt = label['sem_gt_w_bound']
            point_gt = label['point_gt']
            dir_gt = label['reg_dir_gt']
            weight_map = label['loss_weight_map'] if self.if_weighted_loss else None

            loss = dict()

            mask_gt = mask_gt.squeeze(1)
            # dir_gt = dir_gt.squeeze(1)
            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            mask_loss = self._mask_loss(mask_logit, mask_gt, weight_map)
            loss.update(mask_loss)
            # direction branch loss calculation
            dir_loss = self._reg_dir_loss(dir_degree_logit, dir_gt, weight_map)
            loss.update(dir_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, dir_degree_logit, point_logit, mask_gt, dir_gt,
                                                         point_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            seg_logit, dir_map = self.inference(data['img'], metas[0], True)
            seg_pred = seg_logit.argmax(dim=1)
            seg_pred = seg_pred.to('cpu').numpy()
            dir_map = dir_map.to('cpu').numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            dir_map = list(dir_map)
            ret_list = []
            for seg, dir in zip(seg_pred, dir_map):
                ret_dict = {'sem_pred': seg}
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
        sem_logit_list = []
        dir_degree_logit_list = []
        point_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    sem_logit, dir_degree_logit, point_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, dir_degree_logit, point_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dir_degree_logit = self.reverse_tta_transform(dir_degree_logit, rotate_degree, flip_direction)
                point_logit = self.reverse_tta_transform(point_logit, rotate_degree, flip_direction)

                sem_logit = F.softmax(sem_logit, dim=1)
                # dir_degree_logit = dir_degree_logit.sigmoid()

                sem_logit_list.append(sem_logit)
                dir_degree_logit_list.append(dir_degree_logit)
                point_logit_list.append(point_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        point_logit = sum(point_logit_list) / len(point_logit_list)

        dd_map_list = []
        dir_map_list = []
        for idx in range(len(dir_degree_logit_list)):
            dir_degree_logit = dir_degree_logit_list[idx]
            # print(torch.min(dir_degree_logit), torch.max(dir_degree_logit))
            # background = (dir_degree_logit <= 0.3)[0, 0].cpu().numpy()
            dir_degree_logit[dir_degree_logit < 0] = 0
            dir_degree_logit[dir_degree_logit > 2 * np.pi] = 2 * np.pi

            background = (torch.argmax(sem_logit, dim=1)[0] == 0).cpu().numpy()
            # print(dir_degree_logit.shape, background.shape)
            angle_map = dir_degree_logit * 180 / np.pi
            angle_map[angle_map > 180] -= 360
            angle_map = angle_map[0, 0].cpu().numpy()  #[H, W]
            angle_map[background] = 0
            name = meta['file_name'][meta['file_name'].find('/T'):-4]
            # if idx == 0:
            #     print(name)
            #     # id = time.time() % 1000
            #     io.imsave("/root/workspace/NuclearSegmentation/Torch-Image-Segmentation/work_dirs/debug/" + name + '_angle_map.png', angle_map)

            vector_map = angle_to_vector(angle_map, 8)

            dir_map = vector_to_label(vector_map, 8)
            dir_map[background] = -1
            dir_map = dir_map + 1
            # if idx == 0:
            #     io.imsave("/root/workspace/NuclearSegmentation/Torch-Image-Segmentation/work_dirs/debug/" + name + '_dcm_map.png', dir_map / 8 * 255)

            dir_map = torch.from_numpy(dir_map[None, :, :]).cuda()
            # print(type(dir_map), dir_map.shape)
            # dd_map = generate_direction_differential_map(vector_map, self.num_angles + 1, background, True)
            dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)

            # if idx == 0:
            #     io.imsave("/root/workspace/NuclearSegmentation/Torch-Image-Segmentation/work_dirs/debug/" + name + '_ddm_map.png', dd_map[0].cpu().numpy() * 255)

            dir_map_list.append(dir_map)
            dd_map_list.append(dd_map)
        # for dir_logit in dir_logit_list:
        #     dir_logit[:, 0] = dir_logit[:, 0] * sem_logit[:, 0]
        #     dir_map = torch.argmax(dir_logit, dim=1)
        #     dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)
        #     dir_map_list.append(dir_map)
        #     dd_map_list.append(dd_map)

        dd_map = sum(dd_map_list) / len(dd_map_list)

        if self.if_ddm:
            sem_logit = self._ddm_enhencement(sem_logit, dd_map, point_logit)

        return sem_logit, dir_map_list[0]

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

        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        dir_degree_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        point_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch, dir_degree_patch, point_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                dir_degree_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = dir_degree_patch[:, :, ind1_s - i:ind1_e - i,
                                                                                        ind2_s - j:ind2_e - j]
                point_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = point_patch[:, :, ind1_s - i:ind1_e - i,
                                                                              ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        dir_degree_logit = dir_degree_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        point_logit = point_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_degree_logit = resize(dir_degree_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return sem_logit, dir_degree_logit, point_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, dir_degree_logit, point_logit = self.calculate(img)
        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_degree_logit = resize(dir_degree_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return sem_logit, dir_degree_logit, point_logit

    def _mask_loss(self, mask_logit, mask_gt, weight_map=None):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        mask_ce_loss = mask_ce_loss_calculator(mask_logit, mask_gt)
        if weight_map is not None:
            mask_ce_loss *= weight_map[:, 0]
        mask_ce_loss = torch.mean(mask_ce_loss)
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_gt)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _reg_dir_loss(self, dir_degree_logit, dir_gt, weight_map=None):
        dir_loss = {}
        # dir_degree_logit = dir_degree_logit.sigmoid()
        # print(dir_degree_logit.shape, dir_gt.shape)
        dir_mse_loss_calculator = nn.MSELoss()
        dir_degree_mse_loss = dir_mse_loss_calculator(dir_degree_logit, dir_gt)
        dir_loss['dir_degree_mse_loss'] = dir_degree_mse_loss
        return dir_loss

    def _dir_loss(self, dir_logit, dir_gt, weight_map=None):
        dir_loss = {}
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

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, mask_logit, dir_logit, point_logit, mask_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_gt = mask_gt.clone().detach()
        clean_dir_logit = dir_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_gt, self.num_classes)
        # wrap_dict['dir_miou'] = miou(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_gt, self.num_classes)
        # wrap_dict['dir_tiou'] = tiou(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

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
    def _ddm_enhencement(self, mask_logit, dd_map, point_logit):
        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_map = (point_logit / torch.max(point_logit)) > 0.2
        # point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        dd_map = dd_map - (dd_map * point_map)

        # using direction differential map to enhance edge
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] + dd_map) * (1 + dd_map)

        return mask_logit
