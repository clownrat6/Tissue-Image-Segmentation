import torch
import torch.nn as nn


class evolution_area(nn.Module):
    """calcaulate the area of evolution curve."""

    def __init__(self):
        super(evolution_area, self).__init__()

    def forward(self, mask_score, class_weight):
        curve_area = torch.sum(class_weight * mask_score)
        return curve_area


class ActiveContourLoss(nn.Module):

    def __init__(self, area_weight=0.000001, len_weight=10, w_area=False):
        super().__init__()
        self.area_weight = area_weight
        self.len_weight = len_weight
        self.w_area = w_area

    def forward(self, pred, target):
        # length term
        delta_r = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2]**2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        # region term
        c_in = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in = torch.mean(pred * (target - c_in)**2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - pred) * (target - c_out)**2)
        region = region_in + region_out

        loss = self.len_weight * lenth + region

        if self.w_area:
            loss += self.area_weight * torch.sum(pred)

        return loss
