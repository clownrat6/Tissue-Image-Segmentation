import torch

from tiseg.models.backbones import ResNet


def judge_resnet(outs, C, H, W):
    assert len(outs) == 4
    assert outs[0].shape == (2, C * 2**0, H // 4, W // 4)
    assert outs[1].shape == (2, C * 2**1, H // 8, W // 8)
    assert outs[2].shape == (2, C * 2**2, H // 16, W // 16)
    assert outs[3].shape == (2, C * 2**3, H // 32, W // 32)


def judge_deeplab_resnet(outs, C, H, W):
    assert len(outs) == 4
    assert outs[0].shape == (2, C * 2**0, H // 4, W // 4)
    assert outs[1].shape == (2, C * 2**1, H // 8, W // 8)
    assert outs[2].shape == (2, C * 2**2, H // 8, W // 8)
    assert outs[3].shape == (2, C * 2**3, H // 8, W // 8)


def test_resnet():
    base_channels = 64
    N, C, H, W = (2, 3, 224, 224)
    temp = torch.randn((N, C, H, W))
    model = ResNet(
        depth=18,
        in_channels=C,
        base_channels=base_channels,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN'))
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge_resnet(outs, base_channels, H, W)


def test_deeplab_resnet():
    base_channels = 64
    N, C, H, W = (2, 3, 224, 224)
    temp = torch.randn((N, C, H, W))
    model = ResNet(
        depth=18,
        in_channels=C,
        base_channels=base_channels,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 1, 1),
        dilations=(1, 1, 2, 4),
        norm_cfg=dict(type='BN'))
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge_deeplab_resnet(outs, base_channels, H, W)
