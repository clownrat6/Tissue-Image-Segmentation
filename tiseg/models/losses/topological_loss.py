import torch
import torch.nn as nn


class TopologicalLoss(nn.Module):

    def __init__(self, use_regression=True, weight=None):
        super().__init__()
        self.use_regression = use_regression
        self.weight = weight

    def forward(self, pred, target, pred_contour, target_contour):
        if  self.use_regression:
            dir_mse_loss_calculator = nn.MSELoss(reduction='none')
            dir_mse_loss = dir_mse_loss_calculator(pred, target)
            all_contour = pred_contour + target_contour
            loss = torch.sum(dir_mse_loss * all_contour) / torch.sum(all_contour)
        else:
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            dir_ce_loss = dir_ce_loss_calculator(pred, target)
            all_contour = pred_contour + target_contour
            loss = torch.sum(dir_ce_loss * all_contour) / torch.sum(all_contour)
        return loss
