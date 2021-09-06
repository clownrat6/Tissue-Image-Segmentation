import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class UNetEncoderLayer(nn.Module):
    """Encoder stage layer of UNet.

    Usually profile: conv1x1 + conv3x3 + conv3x3 + downsampling, downsampling
    is conv3x3(stride 2) in default.

    Args:
        in_channels (int): The channels of input feature map.
        feedforward_channels (int): The channels of output feature map.
        depth (int): The number of convs.
        norm_cfg (dict): The config of normlization layer.
            Default: dict(type='BN')
        act_cfg (dict): The config of activation layer.
            Default: dict(type='ReLU')

    Returns:
        torch.Tensor: [N, feedforward_channels, H, W]
        torch.Tensor: [N, feedforward_channels, H // 2, W // 2]
    """

    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 depth,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.blocks = []
        # channel projection
        self.blocks.append(
            ConvModule(
                in_channels,
                feedforward_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for i in range(depth - 2):
            self.blocks.append(
                ConvModule(
                    feedforward_channels,
                    feedforward_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.downsampling = ConvModule(
            feedforward_channels,
            feedforward_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        skip = self.blocks(x)
        x = self.downsampling(skip)
        return skip, x


class UNetNeckLayer(nn.Module):
    """Neck layer of UNet.

    Usually profile: conv1x1 + conv3x3 + conv3x3, there is no upsampling
    operation and downsampling operation in this stage.

    Args:
        in_channels (int): The channels of input feature map.
        feedforward_channels (int): The channels of output feature map.
        depth (int): The number of convs.
        norm_cfg (dict): The config of normlization layer.
            Default: dict(type='BN')
        act_cfg (dict): The config of activation layer.
            Default: dict(type='ReLU')

    Returns:
        torch.Tensor: [N, feedforward_channels, H, W]
    """

    def __init__(self,
                 in_channels,
                 feedforward_channels,
                 depth,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.blocks = []
        self.blocks.append(
            ConvModule(
                in_channels,
                feedforward_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        for _ in range(depth - 1):
            self.blocks.append(
                ConvModule(
                    feedforward_channels,
                    feedforward_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class UNetDecoderLayer(nn.Module):
    """Decoder stage layer of UNet.

    Usually profile: upsampling + conv1x1 (process skip connection) + conv3x3,
    upsampling is interpolate + conv1x1 in default.

    Args:
        skip_channels (int): The channels of skip connection feature
            map.
        feedforward_channels (int): The channels of output feature map.
        depth (int): The number of convs.
        in_channels (int, optional): The channels of input feature map.
            Default: None (when in_channels is None, this module will only
            process skip connection feature maps)
        align_corners (bool): Whether to align corners when interpolate.
            Default: False
        norm_cfg (dict): The config of normlization layer.
            Default: dict(type='BN')
        act_cfg (dict): The config of activation layer.
            Default: dict(type='ReLU')

    Returns:
        torch.Tensor: [N, feedforward_channels, H, W]
    """

    def __init__(self,
                 skip_channels,
                 feedforward_channels,
                 depth,
                 in_channels=None,
                 align_corners=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.feedforward_channels = feedforward_channels
        self.depth = depth
        self.align_corners = align_corners
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.blocks = []

        self.skip_only = False
        if self.in_channels is None:
            self.skip_only = True

        # channel projection (only skip or skip concat input)
        if self.skip_only:
            self.blocks.append(
                ConvModule(
                    self.skip_channels,
                    self.feedforward_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            depth += 1
        else:
            self.proj = ConvModule(
                self.in_channels,
                self.feedforward_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            self.blocks.append(
                ConvModule(
                    self.feedforward_channels + self.skip_channels,
                    self.feedforward_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        for _ in range(depth - 2):
            self.blocks.append(
                ConvModule(
                    self.feedforward_channels,
                    self.feedforward_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, skip, x=None):

        if not self.skip_only:
            x = self.proj(x)
            x = F.interpolate(
                x,
                skip.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            x = torch.cat([skip, x], dim=1)
        else:
            x = skip

        out = self.blocks(x)
        return out
