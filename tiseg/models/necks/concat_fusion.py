import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule

from ..builder import NECKS


def generate_spatial_map(H, W, device='cpu'):
    spatial_batch_val = np.zeros((8, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            xmin = w / W * 2 - 1
            xmax = (w + 1) / W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / H * 2 - 1
            ymax = (h + 1) / H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1 / W, 1 / H]
    return torch.tensor(spatial_batch_val).to(device)


@NECKS.register_module()
class ConcatFusion(BaseModule):

    def __init__(self, fusion_only=True):
        super().__init__()
        self.fusion_only = fusion_only

    def forward(self, feats):
        img_feats, txt_feats = feats
        if self.fusion_only:
            out_feats = []
        else:
            out_feats = list(img_feats)
        # l2 normalize of features
        # img_feats: [N, C, H, W]
        # txt_feats: [N, C]
        img_feats = F.normalize(img_feats[-1], p=2, dim=1)
        txt_feats = F.normalize(txt_feats[0], p=2, dim=1)

        N, _, H, W = img_feats.shape
        # txt features broadcast
        # [N, C] -> [N, C, H, W]
        txt_feats = txt_feats.reshape(N, -1, 1, 1).expand(-1, -1, H, W)
        # Fusion
        out = torch.cat([img_feats, txt_feats], dim=1)
        out_feats.append(out)
        return out_feats


@NECKS.register_module()
class SpatialConcatFusion(BaseModule):

    def __init__(self, fusion_only=True):
        super().__init__()
        self.fusion_only = fusion_only

    def forward(self, feats):
        img_feats, txt_feats = feats
        if self.fusion_only:
            out_feats = []
        else:
            out_feats = list(img_feats)
        # l2 normalize of features
        # img_feats: [N, C, H, W]
        # txt_feats: [N, C]
        img_feats = F.normalize(img_feats[-1], p=2, dim=1)
        txt_feats = F.normalize(txt_feats[0], p=2, dim=1)
        # Get spatial map
        N, _, H, W = img_feats.shape
        spatial_map = generate_spatial_map(H, W, img_feats.device)
        # spatial_map broadcast
        # [8, H, W] -> [N, 8, H, W]
        spatial_map = spatial_map.expand(N, *spatial_map.shape)
        # txt features broadcast
        # [N, C] -> [N, C, H, W]
        txt_feats = txt_feats.reshape(N, -1, 1, 1).expand(-1, -1, H, W)
        # Fusion
        out = torch.cat([img_feats, txt_feats, spatial_map], dim=1)
        out_feats.append(out)
        return out_feats
