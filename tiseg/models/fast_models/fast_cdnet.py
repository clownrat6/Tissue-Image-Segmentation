import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads.fast_cd_head import CDHead
from ..losses import GeneralizedDiceLoss, miou, tiou
from ..utils import generate_direction_differential_map


@SEGMENTORS.register_module()
class CDNetSegmentor(BaseModule):
    """Base class for segmentors."""

    def __init__(self, train_cfg, test_cfg):
        super(CDNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = 3
        self.num_angles = 8

        self.backbone = TorchVGG16BN(
            in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4])
        self.head = CDHead(
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            in_dims=(64, 128, 256, 512, 512),
            stage_dims=[64, 128, 256, 512, 512],
            dropout_rate=0.1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            align_corners=False)

    def forward(self, data, label=None, train_cfg=None, test_cfg=None):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        img_feats = self.backbone(data['img'])
        mask_logit, direction_logit, point_logit = self.head(img_feats)

        if self.training:
            assert label is not None
            assert train_cfg is not None
            mask_label = label['gt_semantic_map']
            point_label = label['gt_point_map']
            direction_label = label['gt_direction_map']

            loss = dict()
            mask_logit = resize(input=mask_logit, size=mask_label.shape[2:])
            direction_logit = resize(
                input=direction_logit, size=direction_label.shape[2:])
            point_logit = resize(input=point_logit, size=point_label.shape[2:])

            mask_label = mask_label.squeeze(1)
            direction_label = direction_label.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            mask_loss = self._mask_loss(mask_logit, mask_label)
            loss.update(mask_loss)
            # direction branch loss calculation
            direction_loss = self._direction_loss(direction_logit,
                                                  direction_label)
            loss.update(direction_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_label)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(
                mask_logit, direction_logit, point_logit, mask_label,
                direction_label, point_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert test_cfg is not None

            use_ddm = test_cfg.get('use_ddm', False)
            if use_ddm:
                # The whole image is too huge. So we use slide inference in
                # default.
                mask_logit = resize(
                    input=mask_logit, size=test_cfg['plane_size'])
                direction_logit = resize(
                    input=direction_logit, size=test_cfg['plane_size'])
                point_logit = resize(
                    input=point_logit, size=test_cfg['plane_size'])

                mask_logit = self._ddm_enhencement(mask_logit, direction_logit,
                                                   point_logit)

            return mask_logit

    def _mask_loss(self, mask_logit, mask_label):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = GeneralizedDiceLoss(
            num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(
            mask_ce_loss_calculator(mask_logit, mask_label))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_label)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _direction_loss(self, direction_logit, direction_label):
        direction_loss = {}
        direction_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        direction_dice_loss_calculator = GeneralizedDiceLoss(
            num_classes=self.num_angles + 1)
        direction_ce_loss = torch.mean(
            direction_ce_loss_calculator(direction_logit, direction_label))
        direction_dice_loss = direction_dice_loss_calculator(
            direction_logit, direction_label)
        # loss weight
        alpha = 1
        beta = 1
        direction_loss['direction_ce_loss'] = alpha * direction_ce_loss
        direction_loss['direction_dice_loss'] = beta * direction_dice_loss

        return direction_loss

    def _point_loss(self, point_logit, point_label):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_label)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, mask_logit, direction_logit, point_logit,
                         mask_label, direction_label, point_label):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()
        clean_direction_logit = direction_logit.clone().detach()
        clean_direction_label = direction_label.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)
        wrap_dict['direction_miou'] = miou(clean_direction_logit,
                                           clean_direction_label,
                                           self.num_angles + 1)

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)
        wrap_dict['direction_tiou'] = tiou(clean_direction_logit,
                                           clean_direction_label,
                                           self.num_angles + 1)

        # metric calculate
        mask_pred = torch.argmax(
            mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        mask_pred[mask_pred == (self.num_classes - 1)] = 0
        mask_target = mask_label.cpu().numpy().astype(np.uint8)
        mask_target[mask_target == (self.num_classes - 1)] = 0

        N = mask_pred.shape[0]
        wrap_dict['aji'] = 0.
        for i in range(N):
            aji_single_image = aggregated_jaccard_index(
                mask_pred[i], mask_target[i])
            wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # distributed environment requires cuda tensor
        wrap_dict['aji'] /= N
        wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict

    @classmethod
    def _ddm_enhencement(self, mask_logit, direction_logit, point_logit):
        # make direction differential map
        direction_map = torch.argmax(direction_logit, dim=1)
        direction_differential_map = generate_direction_differential_map(
            direction_map, 9)

        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_logit = point_logit - torch.min(point_logit) / (
            torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        direction_differential_map[point_logit > 0.2] = 0

        # using direction differential map to enhance edge
        mask_logit = F.softmax(mask_logit, dim=1)
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] +
                                   direction_differential_map) * (
                                       1 + 2 * direction_differential_map)

        return mask_logit
