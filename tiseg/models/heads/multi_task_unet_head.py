import torch.nn as nn

from .unet_head import conv3x3, conv1x1, UNetLayer


class MultiTaskBranches(nn.Module):

    def __init__(self, in_dims, feed_dims, num_classes, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.norm_cfg = norm_cfg

        # NOTE: inplace wise relu can largely save gpu memory cost.
        real_act_cfg = dict()
        real_act_cfg['inplace'] = True
        real_act_cfg.update(act_cfg)
        self.act_cfg = real_act_cfg

        num_convs = 3

        self.ms_convs = nn.ModuleList()
        for num_class in num_classes:
            ms_conv = []
            for idx in range(num_convs):
                if idx == 0:
                    ms_conv.append(conv3x3(in_dims, feed_dims, self.norm_cfg, self.act_cfg))
                elif idx == num_convs - 1:
                    ms_conv.append(conv1x1(feed_dims, num_class))
                else:
                    ms_conv.append(conv3x3(feed_dims, feed_dims, self.norm_cfg, self.act_cfg))
            self.ms_convs.append(nn.Sequential(*ms_conv))

    def forward(self, x):
        outs = []
        for ms_conv in self.ms_convs:
            outs.append(ms_conv(x))

        return outs


class MultiTaskUNetHead(nn.Module):
    """UNet for nulcie segmentation task.

    Args:
        stage_convs (list[int]): The conv number of each stage.
            Default: [3, 3, 3, 3]
        stage_channels (list[int]): The feedforward channel number of each
            stage. Default: [16, 32, 64, 128]
    """

    def __init__(self,
                 num_classes,
                 mt_dims=64,
                 bottom_in_dim=512,
                 skip_in_dims=[64, 128, 256, 512, 512],
                 stage_dims=[16, 32, 64, 128, 256],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        assert isinstance(num_classes, list)

        self.num_classes = num_classes
        self.bottom_in_dim = bottom_in_dim
        self.skip_in_dims = skip_in_dims
        self.stage_dims = stage_dims
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        num_layers = len(self.skip_in_dims)

        self.decode_layers = nn.ModuleList()
        for idx in range(num_layers - 1, -1, -1):
            if idx == num_layers - 1:
                # bottom, initial layer
                self.decode_layers.append(
                    UNetLayer(self.bottom_in_dim, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))
            else:
                self.decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                              act_cfg))

        self.postprocess = MultiTaskBranches(self.stage_dims[0], mt_dims, num_classes)

    def forward(self, bottom_input, skip_inputs):
        # decode stage feed forward
        x = bottom_input
        skips = skip_inputs[::-1]

        decode_layers = self.decode_layers
        for skip, decode_stage in zip(skips, decode_layers):
            x = decode_stage(x, skip)

        out = self.postprocess(x)

        return out
