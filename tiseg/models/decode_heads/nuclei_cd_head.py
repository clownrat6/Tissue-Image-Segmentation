import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import HEADS
from ..losses import GeneralizedDiceLoss, miou, tiou
from ..utils import UNetDecoderLayer, generate_direction_differential_map
from .nuclei_decode_head import NucleiBaseDecodeHead


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
        num_classes (int): The number of mask semantic classes.
        num_angles (int): The number of angle types. Default: 8
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 num_classes,
                 num_angles=8,
                 dropout_rate=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.feedforward_channels = feedforward_channels
        self.num_classes = num_classes
        self.num_angles = num_angles
        self.dropout_rate = dropout_rate

        self.mask_pre_branch = RU(self.in_channels, self.feedforward_channels,
                                  norm_cfg, act_cfg)
        self.direction_pre_branch = RU(self.feedforward_channels,
                                       self.feedforward_channels, norm_cfg,
                                       act_cfg)
        self.point_pre_branch = RU(self.feedforward_channels,
                                   self.feedforward_channels, norm_cfg,
                                   act_cfg)

        # Cross Branch Attention
        self.point_to_direction_attention = AU(1)
        self.direction_to_mask_attention = AU(self.num_angles + 1)

        # Prediction Operations
        # dropout will be closed automatically when .eval()
        self.dropout = nn.Dropout2d(self.dropout_rate)
        self.point_pred_op = nn.Conv2d(
            self.feedforward_channels, 1, kernel_size=1)
        self.direction_pred_op = nn.Conv2d(
            self.feedforward_channels, self.num_angles + 1, kernel_size=1)
        self.mask_pred_op = nn.Conv2d(
            self.feedforward_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        mask_feature = self.mask_pre_branch(x)
        direction_feature = self.direction_pre_branch(mask_feature)
        point_feature = self.point_pre_branch(direction_feature)

        # point branch
        point_feature = self.dropout(point_feature)
        point_logit = self.point_pred_op(point_feature)

        # direction branch
        direction_feature_with_point_logit = self.point_to_direction_attention(
            direction_feature, point_logit)
        direction_feature_with_point_logit = self.dropout(
            direction_feature_with_point_logit)
        direction_logit = self.direction_pred_op(
            direction_feature_with_point_logit)

        # mask branch
        mask_feature_with_direction_logit = self.direction_to_mask_attention(
            mask_feature, direction_logit)
        mask_feature_with_direction_logit = self.dropout(
            mask_feature_with_direction_logit)
        mask_logit = self.mask_pred_op(mask_feature_with_direction_logit)

        return mask_logit, direction_logit, point_logit


@HEADS.register_module()
class NucleiCDHead(NucleiBaseDecodeHead):
    """CDNet: Centripetal Direction Network for Nuclear Instance Segmentation

    This head is the implementation of `CDNet <->`_.

    Only support nuclei segmentation now.

    Args:
        stage_convs (list[int]): The number of convolutions of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channels of each stage.
            Default: [16, 32, 64, 128]
    """

    def __init__(self,
                 num_angles=8,
                 stage_convs=[3, 3, 3, 3],
                 stage_channels=[16, 32, 64, 128],
                 **kwargs):
        super().__init__(**kwargs)
        self.num_angles = num_angles
        self.stage_channels = stage_channels
        self.stage_convs = stage_convs

        # initial check
        assert len(self.in_channels) == len(self.in_index) == len(
            self.stage_channels)
        num_stages = len(self.in_channels)

        # judge if the num_stages is valid
        assert num_stages in [
            4, 5
        ], 'Only support four stage or four stage with an extra stage now.'

        # make channel pair
        self.stage_channels.append(None)
        channel_pairs = [(self.in_channels[idx], self.stage_channels[idx],
                          self.stage_channels[idx + 1])
                         for idx in range(num_stages)]
        channel_pairs = channel_pairs[::-1]

        self.decode_stages = nn.ModuleList()
        for (skip_channels, feedforward_channels,
             in_channels), depth in zip(channel_pairs, self.stage_convs):
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
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            dropout_rate=self.dropout_rate,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # decode stage feed forward
        x = None
        skips = inputs[::-1]
        for skip, decode_stage in zip(skips, self.decode_stages):
            x = decode_stage(skip, x)

        # post process
        out = self.post_process(x)

        # CDNet has three branches
        mask_out, direction_out, point_out = out

        return mask_out, direction_out, point_out

    def forward_train(self, inputs, metas, label, train_cfg):
        """Forward function when training phase.

        Args:
            inputs (list[torch.tensor]): Feature maps from backbone.
            metas (list[dict]): Meta information.
            label (dict[torch.tensor]): Ground Truth wrap dict.
                (label usaually contains `gt_semantic_map_with_edge`,
                `gt_point_map`, `gt_direction_map`)
            train_cfg (dict): The cfg of training progress.
        """
        mask_logit, direction_logit, point_logit = self.forward(inputs)

        mask_label = label['gt_semantic_map_with_edge']
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

        # TODO: Conside to remove some edge loss value.
        # mask branch loss calculation
        mask_loss = self._mask_loss(mask_logit, mask_label)
        loss.update(mask_loss)
        # direction branch loss calculation
        direction_loss = self._direction_loss(direction_logit, direction_label)
        loss.update(direction_loss)
        # point branch loss calculation
        point_loss = self._point_loss(point_logit, point_label)
        loss.update(point_loss)

        # calculate training metric
        training_metric_dict = self._training_metric(mask_logit,
                                                     direction_logit,
                                                     point_logit, mask_label,
                                                     direction_label,
                                                     point_label)
        loss.update(training_metric_dict)

        return loss

    def _mask_loss(self, mask_logit, mask_label):
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

    def _point_loss(self, point_logit, point_label):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_label)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _direction_loss(self, direction_logit, direction_label):
        direction_loss = {}
        direction_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        direction_dice_loss_calculator = GeneralizedDiceLoss(
            num_classes=self.num_angles + 1)
        direction_ce_loss = torch.mean(
            direction_ce_loss_calculator(direction_logit, direction_label))
        direction_dice_loss = direction_dice_loss_calculator(
            direction_logit, direction_label)
        # loss weight
        alpha = 1
        beta = 1
        direction_loss['direction_ce_loss'] = alpha * direction_ce_loss
        direction_loss['direction_dice_loss'] = beta * direction_dice_loss

        return direction_loss

    def _training_metric(self, mask_logit, direction_logit, point_logit,
                         mask_label, direction_label, point_label):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()
        clean_direction_logit = direction_logit.clone().detach()
        clean_direction_label = direction_label.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)
        wrap_dict['direction_miou'] = miou(clean_direction_logit,
                                           clean_direction_label,
                                           self.num_angles + 1)
        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)

        wrap_dict['direction_tiou'] = tiou(clean_direction_logit,
                                           clean_direction_label,
                                           self.num_angles + 1)

        # metric calculate
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

    def forward_test(self, inputs, metas, test_cfg):
        # retrieval mask_out / direction_out / point_out
        mask_logit, direction_logit, point_logit = self.forward(inputs)

        use_ddm = test_cfg.get('use_ddm', False)
        if use_ddm:
            # The whole image is too huge. So we use slide inference in
            # default.
            mask_logit = resize(
                input=mask_logit,
                size=test_cfg['plane_size'],
                mode='bilinear',
                align_corners=self.align_corners)
            direction_logit = resize(
                input=direction_logit,
                size=test_cfg['plane_size'],
                mode='bilinear',
                align_corners=self.align_corners)
            point_logit = resize(
                input=point_logit,
                size=test_cfg['plane_size'],
                mode='bilinear',
                align_corners=self.align_corners)

            # make direction differential map
            direction_map = torch.argmax(direction_logit, dim=1)
            direction_differential_map = generate_direction_differential_map(
                direction_map, 9)

            # using point map to remove center point direction differential map
            point_logit = point_logit[:, 0, :, :]
            point_logit = point_logit - torch.min(point_logit) / (
                torch.max(point_logit) - torch.min(point_logit))

            # mask out some redundant direction differential
            direction_differential_map[point_logit > 0.2] = 0

            # using direction differential map to enhance edge
            mask_logit = F.softmax(mask_logit, dim=1)
            mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] +
                                       direction_differential_map) * (
                                           1 + 2 * direction_differential_map)

        return mask_logit
