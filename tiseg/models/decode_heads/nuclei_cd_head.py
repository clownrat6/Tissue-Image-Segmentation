import numpy as np
import torch
import torch.nn as nn

from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import HEADS
from ..losses import GeneralizedDiceLoss, miou, tiou
from .cd_head import CDHead


@HEADS.register_module()
class NucleiCDHead(CDHead):
    """CDNet: Centripetal Direction Network for Nuclear Instance Segmentation

    This head is the implementation of `CDNet <->`_.

    Args:
        num_angles (int): The angle number of direction map. Default: 8
        stage_convs (list[int]): The number of convolutions of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channels of each stage.
            Default: [16, 32, 64, 128]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        wrap_dict['aji'] /= N

        return wrap_dict
