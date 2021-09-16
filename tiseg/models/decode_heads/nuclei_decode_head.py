import numpy as np
import torch
import torch.nn as nn

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import HEADS
from ..losses import GeneralizedDiceLoss, mdice, miou


# TODO: Add doc string & Add comments.
@HEADS.register_module()
class NucleiBaseDecodeHead(nn.Module):
    """"""

    def __init__(self,
                 in_channels,
                 num_classes,
                 in_index=-1,
                 dropout_rate=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 input_transform='multiple_select',
                 align_corners=False):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.in_index = in_index
        self.dropout_rate = dropout_rate
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.input_transform
        self.align_corners = align_corners

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

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
        mask_logit = self.forward(inputs)
        mask_label = label['gt_semantic_map_with_edge']

        loss = dict()
        mask_logit = resize(
            input=mask_logit,
            size=mask_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        mask_label = mask_label.squeeze(1)
        mask_loss = self._mask_loss(mask_logit, mask_label)
        loss.update(mask_loss)

        # calculate training metric
        training_metric_dict = self._training_metric(mask_logit, mask_label)
        loss.update(training_metric_dict)

        return loss

    def _mask_loss(self, mask_logit, mask_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = GeneralizedDiceLoss(num_classes=3)
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

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()
        wrap_dict['mask_dice'] = mdice(clean_mask_logit, clean_mask_label, 3)
        wrap_dict['mask_iou'] = miou(clean_mask_logit, clean_mask_label, 3)

        # metric calculate
        mask_pred = (torch.argmax(mask_logit,
                                  dim=1) == 1).cpu().numpy().astype(np.uint8)
        mask_target = (mask_label == 1).cpu().numpy().astype(np.uint8)

        N = mask_pred.shape[0]
        wrap_dict['aji'] = 0.
        for i in range(N):
            aji_single_image = aggregated_jaccard_index(
                mask_pred[i], mask_target[i])
            wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        wrap_dict['aji'] /= N

        return wrap_dict

    def forward_test(self, inputs, metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)
