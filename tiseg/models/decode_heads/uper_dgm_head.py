import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from tiseg.utils import resize
from ..builder import HEADS
from ..losses import miou, tiou
from ..utils import generate_direction_differential_map
from .cd_head import DGM
from .decode_head import BaseDecodeHead


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, norm_cfg, act_cfg,
                 align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class UPerDGMHead(BaseDecodeHead):
    """UperNet main structure with DGM module of CDNet.

    UPerNet: `<https://arxiv.org/abs/1807.10221>`.
    CDNet: ``.

    Args:
        channels (int): Input channels of postprocess module.
        num_angles (int): The type number of DGM angles. Default: 8
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6)
    """

    def __init__(self,
                 channels,
                 num_angles=8,
                 pool_scales=(1, 2, 3, 6),
                 **kwargs):
        super(UPerDGMHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.channels = channels
        self.num_angles = num_angles
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # custom cls_seg module rather than unified cls_seg from BaseDecodeHead
        if self.dropout_rate > 0:
            self.postprocess = DGM(self.channels, self.channels // 4,
                                   self.num_classes, self.num_angles,
                                   self.dropout_rate, self.norm_cfg,
                                   self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.postprocess(output)
        return output

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

        mask_label = label['gt_semantic_map']
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
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(
            mask_ce_loss_calculator(mask_logit, mask_label))
        # loss weight
        alpha = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss

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
        direction_ce_loss = torch.mean(
            direction_ce_loss_calculator(direction_logit, direction_label))
        # loss weight
        alpha = 1
        direction_loss['direction_ce_loss'] = alpha * direction_ce_loss

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
