import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GeneralizedDiceLoss(nn.Module):
    """This Generalized Dice Loss (for Multi-Class) implementation refer to:

    "Generalised Dice Overlap as a Deep Learning Loss Function for Highly
    Unbalanced Segmentations" - `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7610921/` # noqa
    """

    def __init__(self, num_classes):
        super(GeneralizedDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self,
                logit,
                target,
                smooth=1e-4,
                spatial_weight=None,
                channel_weight=None):
        # one-hot encoding for target
        target_one_hot = _convert_to_one_hot(target, self.num_classes).permute(
            0, 3, 1, 2).contiguous()
        # softmax for logit
        logit = F.softmax(logit, dim=1)

        if spatial_weight is not None:
            logit *= spatial_weight
            target_one_hot *= spatial_weight

        intersect = torch.sum(logit * target_one_hot, dim=(0, 2, 3))
        logit_area = torch.sum(logit, dim=(0, 2, 3))
        target_area = torch.sum(target_one_hot, dim=(0, 2, 3))
        addition_area = logit_area + target_area

        if channel_weight is not None:
            intersect *= channel_weight
            addition_area *= channel_weight

        generalized_dice_score = (2 * torch.sum(intersect) + smooth) / (
            torch.sum(addition_area) + smooth)

        generalized_dice_loss = 1 - generalized_dice_score

        return generalized_dice_loss


class MultiClassDiceLoss(nn.Module):
    """Calculate each class dice loss, then sum per class dice loss as a total
    loss."""

    def __init__(self, num_classes):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logit, target, weights=None):
        assert target.ndim == 3
        # one-hot encoding for target
        target_one_hot = _convert_to_one_hot(target, self.num_classes).permute(
            0, 3, 1, 2).contiguous()
        smooth = 1e-4
        # softmax for logit
        logit = F.softmax(logit, dim=1)

        N, C, _, _ = target_one_hot.shape

        loss = 0

        for i in range(C):
            logit_per_class = logit[:, i]
            target_per_class = target_one_hot[:, i]

            intersection = logit_per_class * target_per_class
            # calculate per class dice loss
            dice_loss_per_class = 2 * (intersection.sum((-2, -1)) + smooth) / (
                logit_per_class.sum((-2, -1)) + target_per_class.sum(
                    (-2, -1)) + smooth)
            dice_loss_per_class = 1 - dice_loss_per_class.sum() / N
            if weights is not None:
                dice_loss_per_class *= weights[i]
            loss += dice_loss_per_class

        return loss


class DiceLoss(nn.Module):
    """The Plain Dice Loss.

    This loss comes from V-Net:
    "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" - # noqa
    `https://arxiv.org/abs/1606.04797`
    """

    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logit, target):
        target_one_hot = _convert_to_one_hot(target, self.num_classes).permute(
            0, 3, 1, 2).contiguous()
        smooth = 1e-4
        # softmax for logit
        logit = F.softmax(logit, dim=1)

        intersect = torch.sum(logit * target_one_hot, dim=(0, 2, 3))
        logit_area = torch.sum(logit, dim=(0, 2, 3))
        target_area = torch.sum(target_one_hot, dim=(0, 2, 3))
        addition_area = logit_area + target_area

        dice_score = torch.mean(
            (2 * intersect + smooth) / (addition_area + smooth))

        dice_loss = 1 - dice_score

        return dice_loss
