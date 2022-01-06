import torch
import torch.nn as nn


class TopologicalLoss(nn.Module):

    def __init__(self, use_regression=True, weight=False, num_angles=None):
        super().__init__()
        self.use_regression = use_regression
        self.weight = weight
        self.num_angles = num_angles

    def forward(self, pred, target, pred_contour, target_contour):
        if  self.use_regression:
            dir_mse_loss_calculator = nn.MSELoss(reduction='none')
            dir_mse_loss = dir_mse_loss_calculator(pred, target)
            all_contour = pred_contour + target_contour
            loss = torch.sum(dir_mse_loss * all_contour) / torch.sum(all_contour)
        else:
            dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
            dir_ce_loss = dir_ce_loss_calculator(pred, target) 
            if self.weight:
                pred_dir = torch.argmax(pred, dim=1)
                diff = torch.abs(pred_dir - target)
                weight = diff.min(self.num_angles - diff)
                background = (pred_dir == 0) + (target == 0)
                weight[background > 0] = 2 
                dir_ce_loss = dir_ce_loss * weight
            all_contour = pred_contour + target_contour
            loss = torch.sum(dir_ce_loss * all_contour) / torch.sum(all_contour)
        return loss
