import torch

from tiseg.models.backbones import VGG


def judge(outs, C, H, W):
    assert len(outs) == 6
    assert outs[0].shape == (2, C * 2**0, H // 2, W // 2)
    assert outs[1].shape == (2, C * 2**1, H // 4, W // 4)
    assert outs[2].shape == (2, C * 2**2, H // 8, W // 8)
    assert outs[3].shape == (2, C * 2**3, H // 16, W // 16)
    assert outs[4].shape == (2, C * 2**3, H // 32, W // 32)
    assert outs[5].shape == (2, 1000, H // 32, W // 32)


def test_vgg():
    base_channels = 64
    N, C, H, W = (2, 3, 383, 384)
    temp = torch.randn((N, C, H, W))
    model = VGG(
        depth=16,
        in_channels=C,
        base_channels=base_channels,
        with_fc=True,
        out_indices=(0, 1, 2, 3, 4, 5),
        norm_cfg=dict(type='BN'))
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge(outs, base_channels, H, W)
