import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss, mdice, tdice
from .base import BaseSegmentor


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvLayer, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                groups=groups))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


# --- different types of layers --- #
class BasicLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BasicLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BottleneckLayer, self).__init__()

        inter_planes = growth_rate * 4
        self.conv1 = ConvLayer(in_channels, inter_planes, kernel_size=1, padding=0)
        self.conv2 = ConvLayer(inter_planes, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# --- dense block structure --- #
class DenseBlock(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, layer_type, dilations):
        super(DenseBlock, self).__init__()
        for i in range(len(dilations)):
            layer = layer_type(in_channels + i * growth_rate, growth_rate, drop_rate, dilations[i])
            self.add_module('denselayer{:d}'.format(i + 1), layer)


def choose_hybrid_dilations(n_layers, dilation_schedule, is_hybrid):
    import numpy as np
    # key: (dilation, n_layers)
    HD_dict = {
        (1, 4): [1, 1, 1, 1],
        (2, 4): [1, 2, 3, 2],
        (4, 4): [1, 2, 5, 9],
        (8, 4): [3, 7, 10, 13],
        (16, 4): [13, 15, 17, 19],
        (1, 6): [1, 1, 1, 1, 1, 1],
        (2, 6): [1, 2, 3, 1, 2, 3],
        (4, 6): [1, 2, 3, 5, 6, 7],
        (8, 6): [2, 5, 7, 9, 11, 14],
        (16, 6): [10, 13, 16, 17, 19, 21]
    }

    dilation_list = np.zeros((len(dilation_schedule), n_layers), dtype=np.int32)

    for i in range(len(dilation_schedule)):
        dilation = dilation_schedule[i]
        if is_hybrid:
            dilation_list[i] = HD_dict[(dilation, n_layers)]
        else:
            dilation_list[i] = [dilation for k in range(n_layers)]

    return dilation_list


@SEGMENTORS.register_module()
class FullNet(BaseSegmentor):

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(FullNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        layer_type = BasicLayer

        n_layers = 6
        in_channels = 24
        dilations = (1, 2, 4, 8, 16, 4, 1)
        growth_rate = 24
        compress_ratio = 0.5
        drop_rate = 0.1

        # 1st conv before any dense block
        self.conv1 = ConvLayer(3, 24, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers=n_layers, dilation_schedule=dilations, is_hybrid=True)

        for i in range(n_blocks):  # no trans in last block
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block%d' % (i + 1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans%d' % (i + 1), trans)
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def calculate(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        return x

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            mask_logit = self.calculate(data['img'])
            assert label is not None
            mask_label = label['sem_gt_w_bound']
            loss = dict()
            mask_label = mask_label.squeeze(1)
            mask_loss = self._mask_loss(mask_logit, mask_label)
            loss.update(mask_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, mask_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            seg_logit = self.inference(data['img'], metas[0], True)
            seg_pred = seg_logit.argmax(dim=1)
            # Extract inside class
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            ret_list = []
            for seg in seg_pred:
                ret_list.append({'sem_pred': seg})
            return ret_list

    def _mask_loss(self, mask_logit, mask_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(mask_ce_loss_calculator(mask_logit, mask_label))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_label)
        # loss weight
        alpha = 5
        beta = 0.5
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()

        wrap_dict['mask_tdice'] = tdice(clean_mask_logit, clean_mask_label, self.num_classes)
        wrap_dict['mask_mdice'] = mdice(clean_mask_logit, clean_mask_label, self.num_classes)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # (the edge id is set `self.num_classes - 1` in default)
        # mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        # mask_pred[mask_pred == (self.num_classes - 1)] = 0
        # mask_target = mask_label.cpu().numpy().astype(np.uint8)
        # mask_target[mask_target == (self.num_classes - 1)] = 0

        # N = mask_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(mask_pred[i], mask_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
