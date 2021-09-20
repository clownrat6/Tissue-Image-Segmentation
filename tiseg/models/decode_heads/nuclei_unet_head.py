import torch.nn as nn

from ..builder import HEADS
from ..utils import UNetDecoderLayer
from .nuclei_decode_head import NucleiBaseDecodeHead


# TODO: Add doc string & Add comments
@HEADS.register_module()
class NucleiUNetHead(NucleiBaseDecodeHead):
    """"""

    def __init__(self,
                 stage_convs=[3, 3, 3, 3],
                 stage_channels=[16, 32, 64, 128],
                 **kwargs):
        super().__init__(**kwargs)
        self.stage_convs = stage_convs
        self.stage_channels = stage_channels

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

        self.dropout = nn.Dropout2d(self.dropout_rate),
        self.postprocess = nn.Conv2d(
            stage_channels[0], self.num_classes, kernel_size=1, stride=1)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # decode stage feed forward
        x = None
        skips = inputs[::-1]
        for skip, decode_stage in zip(skips, self.decode_stages):
            x = decode_stage(skip, x)

        out = self.dropout(x)
        out = self.postprocess(out)

        return out
