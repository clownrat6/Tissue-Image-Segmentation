import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from ..utils import ConvLSTMCell
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class RRNHead(BaseDecodeHead):
    """Referring Image Segmentation via Recurrent Refinement Networks.

    This head is implemented of `RRN <https://openaccess.thecvf.com/content_
    cvpr_2018/papers/Li_Referring_Image_Segmentation_CVPR_2018_paper.pdf>`_.

    Args:
        num_recurrents (int): Number of ConvLSTM recurrent. Default: 4.
        recurrent_channels (int): Channels of ConvLSTM feedforward.
            Default: 512.
    """

    def __init__(self, num_recurrents=4, recurrent_channels=512, **kwargs):
        super(RRNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(self.in_channels) == num_recurrents
        assert self.channels == recurrent_channels

        self.recurrent = ConvLSTMCell(
            recurrent_channels, recurrent_channels, kernel_size=1, bias=True)

        self.linear_projs = nn.ModuleList()
        for in_channel in self.in_channels:
            self.linear_projs.append(
                ConvModule(
                    in_channel,
                    recurrent_channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.conv_seg = nn.Sequential(
            ConvModule(
                recurrent_channels,
                recurrent_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            nn.Conv2d(self.channels, self.num_classes, kernel_size=1))

    def forward(self, x):
        feats = self._transform_inputs(x)

        # Input embeddings
        embeddings = []
        for feat, linear_proj in zip(feats, self.linear_projs):
            embeddings.append(linear_proj(feat))

        # c3, c4, c5, fusion -> fusion, c5, c4, c3
        embeddings = embeddings[::-1]
        # The decription of paper is that fusion is the h0 hidden state.
        # So, what's the c0? After checking official code, we find that fusion
        # is actually x0 and (h0, c0) is initialized as zero state.
        # The official implementation code: https://github.com/liruiyu/referseg_rrn/blob/master/LSTM_model_convlstm_p543.py#L138 # noqa
        N, _, H, W = embeddings[0].shape
        hidden_state, memory_state = self.recurrent.init_hidden(N, (H, W))
        hidden_state_list = [hidden_state]
        memory_state_list = [memory_state]
        for embedding in embeddings:
            hidden_state, memory_state = self.recurrent(
                embedding, (hidden_state, memory_state))
            hidden_state_list.append(hidden_state)
            memory_state_list.append(memory_state)
        last_hidden_state = hidden_state_list[-1]
        out = self.cls_seg(last_hidden_state)
        return out
