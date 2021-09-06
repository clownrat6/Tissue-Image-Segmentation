import torch
import torch.nn as nn

from .utils import overlapping_caculation


def dice(pred, target, num_classes, thresh=None):
    """Dice coefficient calculation."""
    return overlapping_caculation(pred, target, 'Dice', num_classes, thresh)


def mdice(pred, target, num_classes, thresh=None):
    """Dice coefficient calculation."""
    return torch.mean(
        overlapping_caculation(pred, target, 'Dice', num_classes, thresh))


class Dice(nn.Module):
    """Dice Coefficient calculation module."""

    def __init__(self, thresh=None):
        """Module to calculate the dice.

        Args:
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate dice coefficient.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return dice(pred, target, self.thresh)
