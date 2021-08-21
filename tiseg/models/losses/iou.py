import torch.nn as nn

from .utils import overlapping_caculation


def iou(pred, target, thresh=None):
    """Intersection and Union calculation."""
    return overlapping_caculation(pred, target, 'IoU', thresh)


class IoU(nn.Module):
    """Intersection and Union calculation module."""

    def __init__(self, thresh=None):
        """Module to calculate the iou.

        Args:
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate iou.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return iou(pred, target, self.thresh)
