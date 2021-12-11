import torch.nn as nn
from mmcv.cnn import build_activation_layer

from .unet_head import UNetLayer, conv1x1, conv3x3


class RU(nn.Module):
    """Residual Unit.

    Residual Unit comprises of:
    (Conv3x3 + BN + ReLU + Conv3x3 + BN) + Identity + ReLU
    ( . ) stands for residual inside block

    Args:
        in_dims (int): The input channels of Residual Unit.
        out_dims (int): The output channels of Residual Unit.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self, in_dims, out_dims, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.act_layer = build_activation_layer(act_cfg)
        self.residual_ops = nn.Sequential(
            conv3x3(in_dims, out_dims, norm_cfg), self.act_layer, conv3x3(out_dims, out_dims, norm_cfg))
        self.identity_ops = nn.Sequential(conv1x1(in_dims, out_dims))

    def forward(self, x):
        res_value = self.residual_ops(x)
        ide_value = self.identity_ops(x)
        out = ide_value + res_value
        return self.act_layer(out)


class AU(nn.Module):
    """Attention Unit.

    This module use (conv1x1 + sigmoid) to generate 0-1 (float) attention mask.

    Args:
        in_dims (int): The input channels of Attention Unit.
        num_masks (int): The number of masks to generate. Default: 1
    """

    def __init__(self, in_dims, num_masks=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims, num_masks, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, signal, gate):
        """Using gate to generate attention map and assign the attention map to
        signal."""
        attention_map = self.conv(gate)
        return signal * (1 + attention_map)


class DGM(nn.Module):
    """Direction-Guided Refinement Module (DGM)

    This module will accept prediction of regular segmentation output. This
    module has three branches:
    (1) Mask Branch;
    (2) Direction Map Branch;
    (3) Point Map Branch;

    When training phrase, these three branches provide mask, direction, point
    supervision, respectively. When testing phrase, direction map and point map
    provide refinement operations.

    Args:
        in_dims (int): The input channels of DGM.
        feed_dims (int): The feedforward channels of DGM.
        num_classes (int): The number of mask semantic classes.
        num_angles (int): The number of angle types. Default: 8
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_dims,
                 feed_dims,
                 num_classes,
                 num_angles=8,
                 dropout_rate=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.dropout_rate = dropout_rate

        self.mask_pre_branch = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
        self.direction_pre_branch = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)
        self.point_pre_branch = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)

        # Cross Branch Attention
        self.point_to_direction_attention = AU(1)
        self.direction_to_mask_attention = AU(self.num_angles + 1)

        # Prediction Operations
        # dropout will be closed automatically when .eval()
        self.dropout = nn.Dropout2d(self.dropout_rate)
        self.point_pred_op = nn.Conv2d(self.feed_dims, 1, kernel_size=1)
        self.direction_pred_op = nn.Conv2d(self.feed_dims, self.num_angles + 1, kernel_size=1)
        self.mask_pred_op = nn.Conv2d(self.feed_dims, self.num_classes, kernel_size=1)

    def forward(self, x):
        mask_feature = self.mask_pre_branch(x)
        direction_feature = self.direction_pre_branch(mask_feature)
        point_feature = self.point_pre_branch(direction_feature)

        # point branch
        point_feature = self.dropout(point_feature)
        point_logit = self.point_pred_op(point_feature)

        # direction branch
        direction_feature_with_point_logit = self.point_to_direction_attention(direction_feature, point_logit)
        direction_feature_with_point_logit = self.dropout(direction_feature_with_point_logit)
        direction_logit = self.direction_pred_op(direction_feature_with_point_logit)

        # mask branch
        mask_feature_with_direction_logit = self.direction_to_mask_attention(mask_feature, direction_logit)
        mask_feature_with_direction_logit = self.dropout(mask_feature_with_direction_logit)
        mask_logit = self.mask_pred_op(mask_feature_with_direction_logit)

        return mask_logit, direction_logit, point_logit


class CDHead(nn.Module):

    def __init__(self,
                 num_classes,
                 num_angles=8,
                 in_dims=[16, 32, 64, 128],
                 stage_dims=[16, 32, 64, 128],
                 dropout_rate=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.in_dims = in_dims
        self.stage_dims = stage_dims
        self.in_index = [i for i in range(len(in_dims))]
        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        num_layers = len(self.in_dims)

        # make channel pair
        self.decode_layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx == num_layers - 1:
                self.decode_layers.append(UNetLayer(self.in_dims[idx], 0, self.stage_dims[idx], 3, norm_cfg, act_cfg))
            else:
                self.decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.in_dims[idx], self.stage_dims[idx], 4, norm_cfg, act_cfg))

        self.postprocess = DGM(
            stage_dims[0],
            stage_dims[0],
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            dropout_rate=self.dropout_rate,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):

        # decode stage feed forward
        x = None
        skips = inputs[::-1]
        decode_layers = self.decode_layers[::-1]
        for skip, decode_stage in zip(skips, decode_layers):
            x = decode_stage(x, skip)

        out = self.postprocess(x)

        return out
