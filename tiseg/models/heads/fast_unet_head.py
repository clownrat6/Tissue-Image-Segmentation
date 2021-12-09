import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(
        in_dims, out_dims, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(
        in_dims, out_dims, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


class UNetLayer(nn.Module):

    def __init__(self, in_dims, skip_dims, feed_dims, num_convs, norm_cfg,
                 act_cfg):
        super().__init__()
        self.in_dims = in_dims
        self.skip_dims = skip_dims
        self.feed_dims = feed_dims

        if skip_dims != 0:
            self.skip_conv = conv1x1(skip_dims, feed_dims, norm_cfg, act_cfg)
            skip_dims = feed_dims

        convs = [conv1x1(skip_dims + in_dims, feed_dims, norm_cfg, act_cfg)]
        for _ in range(num_convs - 1):
            convs.append(conv3x3(feed_dims, feed_dims, norm_cfg, act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, skip=None):
        if x is None:
            x = skip
            skip = None

        if skip is not None:
            skip_shape = skip.shape[-2:]
            skip = self.skip_conv(skip)
            x = F.interpolate(x, size=skip_shape)
            x = torch.cat([skip, x], dim=1)

        out = self.convs(x)
        return out


class UNetHead(nn.Module):
    """UNet for nulcie segmentation task.

    Args:
        stage_convs (list[int]): The conv number of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channel number of each
            stage. Default: [16, 32, 64, 128]
    """

    def __init__(self,
                 num_classes,
                 in_dims=[16, 32, 64, 128],
                 stage_dims=[16, 32, 64, 128],
                 dropout_rate=0.1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
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
                self.decode_layers.append(
                    UNetLayer(self.in_dims[idx], 0, self.stage_dims[idx], 3,
                              norm_cfg, act_cfg))
            else:
                self.decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.in_dims[idx],
                              self.stage_dims[idx], 4, norm_cfg, act_cfg))

        self.dropout = nn.Dropout2d(self.dropout_rate)
        self.postprocess = nn.Conv2d(
            self.stage_dims[0], self.num_classes, kernel_size=1, stride=1)

    def forward(self, inputs):

        # decode stage feed forward
        x = None
        skips = inputs[::-1]
        self.decode_layers = self.decode_layers[::-1]
        for skip, decode_stage in zip(skips, self.decode_layers):
            x = decode_stage(x, skip)

        out = self.dropout(x)
        out = self.postprocess(out)

        return out
