import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint


from .unet_head import UNetHead, conv1x1, conv3x3


class RU(nn.Module):
    """Residual Unit.

    Residual Unit comprises of:
    (Conv3x3 + BN + ReLU + Conv3x3 + BN) + Identity + ReLU
    ( . ) stands for residual inside block

    Args:
        in_dims (int): The input channels of Residual Unit.
        out_dims (int): The output channels of Residual Unit.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self, in_dims, out_dims, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()

        # NOTE: inplace wise relu can largely save gpu memory cost.
        real_act_cfg = dict()
        real_act_cfg['inplace'] = True
        real_act_cfg.update(act_cfg)

        self.act_layer = build_activation_layer(real_act_cfg)
        self.residual_ops = nn.Sequential(
            conv3x3(in_dims, out_dims, norm_cfg), self.act_layer, conv3x3(out_dims, out_dims, norm_cfg))
        self.identity_ops = nn.Sequential(conv1x1(in_dims, out_dims))

    def forward(self, x):
        ide_value = self.identity_ops(x)
        res_value = self.residual_ops(x)
        out = ide_value + res_value
        return self.act_layer(out)


class AU(nn.Module):
    """Attention Unit.

    This module use (conv1x1 + sigmoid) to generate 0-1 (float) attention mask.

    Args:
        in_dims (int): The input channels of Attention Unit.
        num_masks (int): The number of masks to generate. Default: 1
    """

    def __init__(self, in_dims, num_masks=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims, num_masks, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, signal, gate):
        """Using gate to generate attention map and assign the attention map to
        signal."""
        attn_map = self.conv(gate)
        return signal * (1 + attn_map)


class DGM(nn.Module):
    """Direction-Guided Refinement Module (DGM)

    This module will accept prediction of regular segmentation output. This
    module has three branches:
    (1) Mask Branch;
    (2) Direction Map Branch;
    (3) Point Map Branch;

    When training phrase, these three branches provide mask, direction, point
    supervision, respectively. When testing phrase, direction map and point map
    provide refinement operations.

    Args:
        in_dims (int): The input channels of DGM.
        feed_dims (int): The feedforward channels of DGM.
        num_classes (int): The number of mask semantic classes.
        num_angles (int): The number of angle types. Default: 8
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_dims,
                 feed_dims,
                 num_classes,
                 num_angles=8,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.num_classes = num_classes
        self.num_angles = num_angles

        self.mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
        self.dir_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)
        self.point_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)


        # Prediction Operations
        self.point_conv = nn.Conv2d(self.feed_dims, 1, kernel_size=1)
        self.dir_conv = nn.Conv2d(self.feed_dims, self.num_angles + 1, kernel_size=1)
        self.foreground_couv = nn.Conv2d(self.feed_dims, self.num_classes - 1, kernel_size=1)
        self.mask_conv = nn.Conv2d(9, self.num_classes, kernel_size=1)

        # QKV

        
        self.point_Q = nn.Conv2d(1, 9, kernel_size=1)
        self.dir_K = nn.Conv2d(self.num_angles + 1, 9, kernel_size=1)
        self.fore_V = nn.Conv2d(self.num_classes - 1, 9, kernel_size=1)

        # Voronoi Attention
        self.SA = BasicLayer(
                dim=9,
                depth=1,
                num_heads=3,
                window_size=256,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=0,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
        self.norm_layer = nn.LayerNorm(9)



    def forward(self, x):
        mask_feature = self.mask_feats(x)
        dir_feature = self.dir_feats(mask_feature)
        point_feature = self.point_feats(dir_feature)

        # point branch
        point_logit = self.point_conv(point_feature)

        # direction branch
        dir_logit = self.dir_conv(dir_feature)

        # foreground branch
        foreground_logit = self.foreground_couv(mask_feature)

        # voronoi attention
        Voronoi = Get_RC(point_logit.detach().cpu().numpy())
        
        xq = self.point_Q(point_logit)
        xk = self.dir_K(dir_logit)
        xv = self.fore_V(foreground_logit)
        
        mask_feature_with_att, H, W = self.SA(xq, xk, xv, 256, 256, Voronoi)
        mask_feature_with_att = self.norm_layer(mask_feature_with_att)
        mask_feature_with_att = mask_feature_with_att.view(-1, H, W, 9).permute(0, 3, 1, 2).contiguous()
        # mask branch
        mask_logit = self.mask_conv(mask_feature_with_att)

        return mask_logit, dir_logit, point_logit, foreground_logit


class CDVoronoiHead(UNetHead):

    def __init__(self, num_classes, num_angles=8, dgm_dims=64, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.num_classes = num_classes
        self.num_angles = num_angles

        self.postprocess = DGM(
            self.stage_dims[0],
            dgm_dims,
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)







class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xq, xk, xv, mask=None, Voronoi=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = xq.shape
        
        #qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        
        #1 * B_ *  numhead * N * 12
        
        q = self.q(xq).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]

        k = self.q(xk).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = k[0]
        
        v = self.q(xv).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = v[0]
        x = v.detach()
        
        q = q * self.scale
        for b in range(B_):
            cnt = np.max(Voronoi[b])     
            V = Voronoi[b].reshape(256*256)
            for i in range(1, cnt+1):
                p = V == i
                num = p.sum()
                if num > 1500:
                    x[b, :, p, :] = v[b, :, p, :]
                    continue
                qi = q[b , :, p, :]
                ki = k[b , :, p, :]
                attn = (qi @ ki.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#                 if mask is not None:
#                     nW = mask.shape[0]
#                     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#                     attn = attn.view(-1, self.num_heads, N, N)
#                     attn = self.softmax(attn)
#                 else:
                attn = self.softmax(attn)

                attn = self.attn_drop(attn)
                vi = v[b , :, p, :]
                x[b, :, p, :] = (attn @ vi)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, xq, xk, xv, mask_matrix, Voronoi):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = xq.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        
        '''
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        '''
        
        shortcut = xv
        xv = self.norm1(xv)
        xv = xv.view(B, H, W, C)
        
        xq = self.norm1(xq)
        xq = xv.view(B, H, W, C)
        
        xk = self.norm1(xk)
        xk = xv.view(B, H, W, C)
        

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        
        xv = F.pad(xv, (0, 0, pad_l, pad_r, pad_t, pad_b))
        xq = F.pad(xq, (0, 0, pad_l, pad_r, pad_t, pad_b))
        xk = F.pad(xk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        _, Hp, Wp, _ = xv.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_xq = torch.roll(xq, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_xk = torch.roll(xk, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_xv = torch.roll(xv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            
            attn_mask = mask_matrix
        else:
            shifted_xq = xq
            shifted_xk = xk
            shifted_xv = xv
            attn_mask = None

        # partition windows
        xq_windows = window_partition(shifted_xq, self.window_size)  # nW*B, window_size, window_size, C
        xq_windows = xq_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        
        xk_windows = window_partition(shifted_xk, self.window_size)  # nW*B, window_size, window_size, C
        xk_windows = xk_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        xv_windows = window_partition(shifted_xv, self.window_size)  # nW*B, window_size, window_size, C
        xv_windows = xv_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C        
        

        # W-MSA/SW-MSA
        attn_windows = self.attn(xq_windows, xk_windows, xv_windows, attn_mask, Voronoi)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                shift_size = 0,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, xq, xk, xv, H, W, Voronoi):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        xq = xq.flatten(2).transpose(1, 2)
        xk = xk.flatten(2).transpose(1, 2)
        xv = xv.flatten(2).transpose(1, 2)
        
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=xq.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

#         mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_mask = None
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                xv = checkpoint.checkpoint(blk, xq, xk, xv, attn_mask, Voronoi)
                xq = xv
                xk = xv
            else:
                xv = blk(xq, xk, xv, attn_mask, Voronoi)
                xq = xv
                xk = xv
        #if self.downsample is not None:
        #    x_down = self.downsample(x, H, W)
        #    Wh, Ww = (H + 1) // 2, (W + 1) // 2
        #    return x, H, W, x_down, Wh, Ww
        #else:
        #return x, H, W, x, H, W
        return xv, H, W


from numba import jit
import numpy as np

@jit(nopython=True)
def getRC2(tmp, x):
    cnt = np.max(tmp)
#     print('cnt', cnt)
    n = m = 256
#     print(x.dtype)
    RC = np.zeros((256, 256), dtype = np.int16)
    ma = [-10000.1] * (cnt + 1)
    posx = np.zeros((cnt+1), dtype = np.int16)
    posy = np.zeros((cnt+1), dtype = np.int16)
    for i in range(n):
        for j in range(m):
            k = tmp[0][i][j]
            v = x[0][i][j]
            if v > ma[k]:
                ma[k] = v
                posx[k] = i
                posy[k] = j
#     print('ma', ma)
    for i in range(n):
        for j in range(m):
            mi = 10000000000
            for k in range(1, cnt+1):
                d = (posx[k] - i) ** 2 + (posy[k] - j) ** 2
                if d < mi:
                    mi = d
                    RC[i][j] = k   
#     for i in range(1, cnt+1):
#         print((RC == i).sum())
    return RC

def Get_RC(x):
    from skimage import measure
    B = x.shape[0]
    n, m = x.shape[2], x.shape[3]
    RC = np.zeros((B, n, m), dtype = np.int16)
    for b in range(B):
        x[b] = (x[b] - np.min(x[b]))/ np.max(x[b])
        tmp = x[b] > 0.2
        tmp = measure.label(tmp)
        RC[b] = getRC2(tmp, x[b])
    return RC