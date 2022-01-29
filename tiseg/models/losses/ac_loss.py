import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from skimage import io
import numpy as np

class evolution_area(nn.Module):
    """calcaulate the area of evolution curve."""

    def __init__(self):
        super(evolution_area, self).__init__()

    def forward(self, mask_score, class_weight):
        curve_area = torch.sum(class_weight * mask_score)
        return curve_area


class ActiveContourLoss(nn.Module):

    def __init__(self, area_weight=0.000001, len_weight=0, w_area=False, show=False):
        super().__init__()
        self.area_weight = area_weight
        self.len_weight = len_weight
        self.w_area = w_area
        self.show = show

    def forward(self, pred, target):
        # length term
        
        delta_r = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2]**2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        # orign length
        # length = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        # new length

        delta_r_target = target[:, :, 1:, :] - target[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c_target = target[:, :, :, 1:] - target[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r_target = delta_r_target[:, :, 1:, :-2]**2  # (B, C, H-2, W-2)
        delta_c_target = delta_c_target[:, :, :-2, 1:]**2  # (B, C, H-2, W-2)
        delta_target = torch.abs(delta_r_target + delta_c_target)
        
        length = torch.mean((torch.sqrt(delta_pred + epsilon) - torch.sqrt(delta_target + epsilon)) ** 2)



        # region term
        c_in = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in = torch.mean(pred * (target - c_in)**2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - pred) * (target - c_out)**2)
        region = region_in + region_out * 1
        loss = self.len_weight * length + region 
        if self.w_area:
            loss += self.area_weight * torch.sum(pred)
        # if self.show:
        #     random.seed(2022)
        #     visual_in = pred * (target - c_in)**2
        #     visual_out = (1 - pred) * (target - c_out)**2
        #     visual_length = F.pad(torch.sqrt(delta_pred + epsilon), (1, 1, 1, 1), 'constant', 0)

            
        #     save_dir = "/root/workspace/NuclearSegmentation/Torch-Image-Segmentation/AC_visual_0length/"
        #     for i in range(pred.shape[0]):
        #         id = int(random.random() * 10000) % 100
        #         l1 = np.zeros((256, 2), np.uint8)
        #         l1[:, :] = 127
        #         t1 = np.uint8(pred[i, 0].cpu().detach().numpy() * 255)
        #         t2 = np.uint8(target[i, 0].cpu().detach().numpy() * 255)
        #         t3 = np.uint8(visual_in[i, 0].cpu().detach().numpy() * 255)
        #         t4 = np.uint8(visual_out[i, 0].cpu().detach().numpy() * 255)
        #         t5 = np.uint8(visual_length[i, 0].cpu().detach().numpy() * 255)
        #         t_all = np.concatenate((t1, l1, t2, l1, t3, l1, t4, l1, t5), axis = 1)
        #         io.imsave(save_dir + str(id) + '_ACloss.png', t_all)
        #         # io.imsave(save_dir + str(id) + '_visual_in.png', np.uint8(visual_in[i, 0].cpu().detach().numpy() * 255))
        #         # io.imsave(save_dir + str(id) + '_visual_out.png', np.uint8(visual_out[i, 0].cpu().detach().numpy() * 255))
        #         # io.imsave(save_dir + str(id) + '_pred.png', np.uint8(pred[i, 0].cpu().detach().numpy() * 255))
        #         # io.imsave(save_dir + str(id) + '_target.png', np.uint8(target[i, 0].cpu().detach().numpy() * 255))
                
        #         # break
            
        #     print('region_in', region_in)
        #     print('region_out', region_out)
        #     print('length', self.len_weight * length)
        #     print('AC_all', loss)
        # exit(0)
        return loss


class LossVariance(nn.Module):
    """ The instances in target should be labeled 
    """

    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        
        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]
            print(unique_vals)
            sum_var = 0
            for val in unique_vals:
                # print(input[k].shape)
                # print((target[k] == val).shape)
                instance = input[k][:, target[k] == val]
                # print(instance.shape)
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        return loss