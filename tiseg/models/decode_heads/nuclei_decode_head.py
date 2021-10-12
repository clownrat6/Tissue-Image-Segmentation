import numpy as np
import torch
import torch.nn as nn

from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import HEADS
from ..losses import GeneralizedDiceLoss, accuracy, miou, tiou
from .decode_head import BaseDecodeHead


# TODO: Add doc string & Add comments.
@HEADS.register_module()
class NucleiBaseDecodeHead(BaseDecodeHead):
    """Nuclei Segmentation Task Basic Decode Head Class.

    Default Training Loss: CE + Dice loss; Default Training Metric: Aji, tIoU,
    mIoU;
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _mask_loss(self, mask_logit, mask_label):
        """calculate mask branch loss."""
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

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()

        wrap_dict['mask_accuracy'] = accuracy(clean_mask_logit,
                                              clean_mask_label)
        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)
        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)

        # metric calculate (the edge id is set `self.num_classes - 1` in
        # default)
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
