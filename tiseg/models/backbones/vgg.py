import torch.nn as nn
from mmcv.cnn import (Conv2d, ConvModule, Linear, constant_init, kaiming_init,
                      normal_init)
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from tiseg.utils import get_root_logger
from ..builder import BACKBONES


def make_vgg_layer(in_channels,
                   out_channels,
                   num_blocks,
                   norm_cfg=None,
                   act_cfg=dict(type='ReLU'),
                   ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=(3 - 1) // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        layers.append(layer)
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


@BACKBONES.register_module()
class VGG(BaseModule):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        in_channels (int): Number of input image channels. Default: 3.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): VGG stages. Default: 5.
        out_indices (Sequence[int], optional): Output from which stages.
            If only one stage is specified, a single tensor (feature map) is
            returned, otherwise multiple stages are specified, a tuple of
            tensors will be returned. When it is None, the default behavior
            depends on whether num_classes is specified. If num_classes <= 0,
            the default value is (4, ), outputing the last feature map before
            classifier. If num_classes > 0, the default value is (5, ),
            outputing the classification score. Default: None.
        act_cfg (dict): The config control activation layer build.
            Default: dict(type='ReLU').
        norm_cfg (dict, optional): The config control norm layer build.
            Default: None.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        ceil_mode (bool): Whether to use ceil_mode of MaxPool. Default: False.
        with_last_pool (bool): Whether to keep the last pooling before
            classifier. Default: True.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers. Each element specifies the number of conv in
    # each stage. For example, VGG11 contains 11 layers with learnable
    # parameters. 11 is computed as 11 = (1 + 1 + 2 + 2 + 2) + 3,
    # where 3 indicates the last three fully-connected layers.
    arch_settings = {
        11: (1, 1, 2, 2, 2),
        13: (2, 2, 2, 2, 2),
        16: (2, 2, 3, 3, 3),
        19: (2, 2, 4, 4, 4)
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 out_indices=None,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=None,
                 norm_eval=False,
                 ceil_mode=False,
                 with_last_pool=True,
                 pretrained=None,
                 init_cfg=None):
        super(VGG, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'

        # Set model architecture setting by depth.
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for vgg')
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        # Set some arg to control model inference.
        self.norm_eval = norm_eval

        # Set which feature maps need to output.
        if out_indices is None:
            out_indices = (4, )
        assert max(out_indices) < 5
        assert max(out_indices) <= num_stages
        self.out_indices = out_indices

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels = in_channels

        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            end_idx = start_idx + num_blocks + 1
            out_channels = self.base_channels * 2**i if i < 4 else in_channels
            vgg_layer = make_vgg_layer(
                in_channels,
                out_channels,
                num_blocks,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            in_channels = out_channels
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1

        self.features = nn.Sequential(*vgg_layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_init(m.weight)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, _BatchNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, Linear):
                normal_init(m.weight, std=0.01)
                if m.bias is not None:
                    constant_init(m.bias, 0)

        if self.pretrained is not None:
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        outs = []
        vgg_layers = self.features
        for i in range(len(self.stage_blocks)):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)

        return outs

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
