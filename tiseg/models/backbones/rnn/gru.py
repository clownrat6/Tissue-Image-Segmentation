import warnings

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint

from tiseg.utils import get_root_logger
from ...builder import BACKBONES


@BACKBONES.register_module()
class GRU(BaseModule):
    """Wrapper class of torch.nn.GRU.

    We may plan to support:
    * many-to-many;
    * many-to-one;

    The input format is fixed to [N, L] and the model doesn't
    support [L, N] input. (N: batch_size, L: sequence_len)

    The word embedding operation is conducted in backbone model.
    """

    def __init__(self,
                 vocab_size,
                 embed_dims,
                 hidden_channels,
                 num_layers=1,
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 out_mode='many-to-many',
                 concat_embedding=False,
                 with_embedding=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')

        assert out_mode.lower() in [
            'many-to-many', 'many-to-one-concat', 'many-to-one-single'
        ], 'we only support many-to-many and many-to-one output modes.'

        self.out_mode = out_mode
        self.concat_embedding = concat_embedding
        self.with_embedding = with_embedding

        # Arg judge
        flag = (concat_embedding and out_mode.lower() == 'many-to-many') or \
            (not concat_embedding)
        assert flag, \
            'Don\'t support many-to-one and concat_embedding simutaneously.'

        flag = (with_embedding and out_mode.lower() == 'many-to-many') or \
            (not with_embedding)
        assert flag, \
            'Don\'t support many-to-one and with_embedding simutaneously.'

        assert not (concat_embedding and with_embedding), 'with_embedding and '
        'concat_embedding can\'t be set simutaneously.'

        self.word_embedding = nn.Embedding(vocab_size, embed_dims)

        self.gru = nn.GRU(
            input_size=embed_dims,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)

    def init_weights(self):
        if self.pretrained is None:
            super().init_weights()
        elif isinstance(self.pretrained, str):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            self.load_state_dict(state_dict, strict=False)
        else:
            raise NotImplementedError

    def forward(self, x, hx=None):
        """GRU forward function.

        Args:
            x (Tensor): input text of shape [N, L].
            hx (tuple(Tensor)): initial state of cell state and
                hidden state (h0, c0). Default: None. (When hx is set to None,
                initial state will be set to zero.)
        Reture:
            (Tensor | tuple[Tensor]): output hidden states of each loop or
                output hidden state and cell state of last loop.
        """
        # Reduce channel dimension
        x = x[:, 0, :]
        x = self.word_embedding(x)
        out, hn = self.gru(x, hx)
        if self.out_mode == 'many-to-many':
            ret = (out, )
        elif self.out_mode == 'many-to-one-concat':
            hn = torch.cat([x for x in hn], dim=1)
            ret = (hn, )
        elif self.out_mode == 'many-to-one-single':
            ret = (hn[-1], )
        else:
            raise NotImplementedError

        if self.concat_embedding:
            return tuple([torch.cat((x, item), dim=2) for item in ret])
        elif self.with_embedding:
            temp = [x]
            temp.extend(ret)
            return temp
        else:
            return ret
