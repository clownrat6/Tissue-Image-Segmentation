import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import NECKS
from .concat_fusion import generate_spatial_map


@NECKS.register_module()
class RMIFusion(BaseModule):

    def __init__(self,
                 visual_in_channels,
                 visual_proj_channels,
                 linguistic_channels,
                 embedding_channels,
                 m_rnn_channels,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.visual_channels = visual_in_channels
        self.lignuistic_channels = linguistic_channels

        self.proj = nn.Conv2d(
            in_channels=visual_in_channels,
            out_channels=visual_proj_channels,
            kernel_size=1)

        # multi_modal lstm
        self.mlstm = nn.LSTM(
            input_size=visual_proj_channels + linguistic_channels +
            embedding_channels + 8,
            hidden_size=m_rnn_channels,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

    def forward(self, feats):
        img_feats, txt_feats = feats

        # [N, C, H, W]
        img_feats = img_feats[0]
        # [N, L, C] with [N, L, C]
        word_embeddings, lang_feats = txt_feats

        N, _, H, W = img_feats.shape
        _, L, _ = word_embeddings.shape

        # l2 normalize of features
        img_feats = F.normalize(img_feats, p=2, dim=1)
        lang_feats = F.normalize(lang_feats, p=2, dim=1)

        # Channel Projection
        img_feats = self.proj(img_feats)

        # Get spatial map
        spatial_map = generate_spatial_map(H, W, img_feats.device)
        # spatial_map broadcast
        # [8, H, W] -> [N, 8, H, W]
        spatial_map = spatial_map.expand(N, *spatial_map.shape)

        # Feature feats (N, Cv, H, W) concat with Word embeddings (N, Ce, H, W)
        # , Linguistic feats (N, Cl, H, W) and Spatial map (N, 8, H, W)
        for i in range(L):
            # Extract each word semantic info and broadcast to shape of [H, W]
            word_feats = lang_feats[:, i, :]
            word_embedding = word_embeddings[:, i, :]
            word_feats = word_feats.reshape(N, -1, 1, 1).expand(-1, -1, H, W)
            word_embedding = word_embedding.reshape(N, -1, 1,
                                                    1).expand(-1, -1, H, W)
            fusion_feats = torch.cat(
                [img_feats, word_embedding, word_feats, spatial_map], dim=1)

            # Reshape to [N * H * W, 1, Cv + Ce + Cl + 8]
            # [N, L, C] is the acceptable shape of `nn.LSTM`
            fusion_feats = fusion_feats.flatten(2).transpose(1, 2).reshape(
                N * H * W, 1, -1)

            # Multi Modal LSTM forward
            if i == 0:
                out, (hn, cn) = self.mlstm(fusion_feats)
            else:
                out, (hn, cn) = self.mlstm(fusion_feats, (hn, cn))

        # [N * H * W, 1, Cm] -> [N, Cm, H, W] for per-pixel classification
        multi_modal_feats = out.reshape(N, H * W,
                                        -1).transpose(1,
                                                      2).reshape(N, -1, H, W)

        return [multi_modal_feats]
