import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer

from tiseg.utils import resize
from ..builder import HEADS
from ..utils import UNetDecoderLayer, UNetNeckLayer
from .decode_head import BaseDecodeHead


class RU(nn.Module):
    """Residual Unit.

    Residual Unit comprises of:
    (Conv3x3 + BN + ReLU + Conv3x3 + BN) + Identity + ReLU
    ( . ) stands for residual inside block

    Args:
        in_channels (int): The input channels of Residual Unit.
        out_channels (int): The output channels of Residual Unit.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.act_layer = build_activation_layer(act_cfg)
        self.residual_ops = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None), self.act_layer,
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None))
        self.identity_ops = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        res_value = self.residual_ops(x)
        ide_value = self.identity_ops(x)
        out = ide_value + res_value
        return self.act_layer(out)


class AU(nn.Module):
    """Attention Unit.

    This module use (conv1x1 + sigmoid) to generate 0-1 (float) attention mask.

    Args:
        in_channels (int): The input channels of Attention Unit.
        num_masks (int): The number of masks to generate. Default: 1
    """

    def __init__(self, in_channels, num_masks=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_masks, kernel_size=1, bias=False),
            nn.Sigmoid())

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
        in_channels (int): The input channels of DGM.
        feedforward_channels (int): The feedforward channels of DGM.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.mask_pre_branch = RU(in_channels, feedforward_channels, norm_cfg,
                                  act_cfg)
        self.direction_pre_branch = RU(feedforward_channels,
                                       feedforward_channels, norm_cfg, act_cfg)
        self.point_pre_branch = RU(feedforward_channels, feedforward_channels,
                                   norm_cfg, act_cfg)

        # Cross Branch Attention
        self.point_to_direction_attention = AU(1)
        self.direction_to_mask_attention = AU(9)

        # Prediction Operations
        self.point_pred_op = nn.Conv2d(feedforward_channels, 1, kernel_size=1)
        self.direction_pred_op = nn.Conv2d(
            feedforward_channels, 9, kernel_size=1)
        self.mask_pred_op = nn.Conv2d(feedforward_channels, 3, kernel_size=1)

    def forward(self, x):
        mask_feature = self.mask_pre_branch(x)
        direction_feature = self.direction_pre_branch(mask_feature)
        point_feature = self.point_pre_branch(direction_feature)

        # point branch
        point_logit = self.point_pred_op(point_feature)

        # direction branch
        direction_feature_with_point_logit = self.point_to_direction_attention(
            direction_feature, point_logit)
        direction_logit = self.direction_pred_op(
            direction_feature_with_point_logit)

        # mask branch
        mask_feature_with_direction_logit = self.direction_to_mask_attention(
            mask_feature, direction_logit)
        mask_logit = self.mask_pred_op(mask_feature_with_direction_logit)

        return mask_logit, direction_logit, point_logit


@HEADS.register_module()
class CDHead(BaseDecodeHead):
    """CDNet: Centripetal Direction Network for Nuclear Instance Segmentation

    This head is the implementation of `CDNet <->`_.

    Args:
        stage_convs (list[int]): The number of convolutions of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channels of each stage.
            Default: [16, 32, 64, 128]
        extra_stage_channels (int, optional): Set the extra stage channels.
            Default: None
        extra_stage_convs (int, optional): Set the number of extra stage convs.
            Default: None.
    """

    def __init__(self,
                 stage_convs=[3, 3, 3, 3],
                 stage_channels=[16, 32, 64, 128],
                 extra_stage_channels=None,
                 extra_stage_convs=None,
                 **kwargs):
        self.stage_channels = stage_channels
        self.extra_stage_channels = extra_stage_channels
        self.extra_stage_convs = extra_stage_convs
        if extra_stage_channels is None:
            assert extra_stage_convs is None, 'Extra stage can\'t be set.'
        super().__init__(input_transform='multiple_select', **kwargs)

        # initial check
        assert len(self.in_channels) == len(self.in_index) == len(
            self.stage_channels)
        num_stages = len(self.in_channels)

        # make extra stage
        self.with_extra_stage = False
        if self.extra_stage_channels is not None:
            self.stage_channels.append(extra_stage_channels)
            self.extra_downsampling = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1)
            self.extra_stage = UNetNeckLayer(
                self.in_channels[-1],
                self.extra_stage_channels,
                self.extra_stage_convs,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.with_extra_stage = True
        else:
            self.stage_channels.append(None)

        # judge if the num_stages is valid
        assert num_stages in [
            4
        ], 'Only support four stage or four stage with an extra stage now.'

        # make channel pair
        channel_pairs = [(self.in_channels[idx], self.stage_channels[idx],
                          self.stage_channels[idx + 1])
                         for idx in range(num_stages)]
        channel_pairs = channel_pairs[::-1]

        self.decode_stages = nn.ModuleList()
        for (skip_channels, feedforward_channels,
             in_channels), depth in zip(channel_pairs, stage_convs):
            self.decode_stages.append(
                UNetDecoderLayer(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    feedforward_channels=feedforward_channels,
                    depth=depth,
                    align_corners=self.align_corners,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ))

        self.post_process = DGM(
            stage_channels[0],
            stage_channels[0],
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Deprecated cls_seg
        del self.conv_seg

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # extra stage process
        x = inputs[-1]
        if self.with_extra_stage:
            x = self.extra_downsampling(x)
            x = self.extra_stage(x)
        else:
            x = None

        # decode stage feed forward
        skips = inputs[::-1]
        for skip, decode_stage in zip(skips, self.decode_stages):
            x = decode_stage(skip, x)

        # post process
        mask_out, direction_out, point_out = self.post_process(x)

        return mask_out, direction_out, point_out

    def forward_train(self, inputs, metas, label, train_cfg):
        mask_logit, direction_logit, point_logit = self.forward(inputs)

        mask_label = label['gt_semantic_map_edge']
        point_label = label['gt_point_map']
        direction_label = label['gt_direction_map']

        loss = dict()
        mask_logit = resize(
            input=mask_logit,
            size=mask_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        direction_logit = resize(
            input=direction_logit,
            size=direction_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        point_logit = resize(
            input=point_logit,
            size=point_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        mask_label = mask_label.squeeze(1)
        direction_label = direction_label.squeeze(1)

        # mask branch loss calculation
        mask_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_loss = torch.mean(mask_loss_calculator(mask_logit, mask_label))

        # point branch loss calculation
        point_loss_calculator = nn.MSELoss()
        point_loss = point_loss_calculator(point_logit, point_label)

        # direction branch loss calculation
        direction_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        direction_loss = torch.mean(
            direction_loss_calculator(direction_logit, direction_label))

        # TODO: Conside to remove some edge loss value.

        loss['mask_loss'] = mask_loss
        loss['direction_loss'] = direction_loss
        loss['point_loss'] = point_loss

        return loss

    def forward_test(self, ):
        pass
