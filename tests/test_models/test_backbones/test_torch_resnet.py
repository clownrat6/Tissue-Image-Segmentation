import torch

from tiseg.models.backbones import TorchResNet


def judge_resnet(outs, C, H, W):
    assert len(outs) == 5
    assert outs[0].shape == (2, C * 2**0, H // 2, W // 2)
    assert outs[1].shape == (2, C * 2**2, H // 4, W // 4)
    assert outs[2].shape == (2, C * 2**3, H // 8, W // 8)
    assert outs[3].shape == (2, C * 2**4, H // 16, W // 16)
    assert outs[4].shape == (2, C * 2**5, H // 32, W // 32)


def judge_deeplab_resnet(outs, C, H, W):
    assert len(outs) == 5
    assert outs[0].shape == (2, C * 2**0, H // 2, W // 2)
    assert outs[1].shape == (2, C * 2**2, H // 4, W // 4)
    assert outs[2].shape == (2, C * 2**3, H // 8, W // 8)
    assert outs[3].shape == (2, C * 2**4, H // 8, W // 8)
    assert outs[4].shape == (2, C * 2**5, H // 8, W // 8)


def test_resnet():
    base_channels = 64
    N, C, H, W = (2, 3, 224, 224)
    temp = torch.randn((N, C, H, W))
    model = TorchResNet(
        'resnet50-d32', out_indices=(0, 1, 2, 3, 4), pretrained=False)
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge_resnet(outs, base_channels, H, W)


def test_deeplab_resnet():
    base_channels = 64
    N, C, H, W = (2, 3, 224, 224)
    temp = torch.randn((N, C, H, W))
    model = TorchResNet(
        'resnet50-d8', out_indices=(0, 1, 2, 3, 4), pretrained=False)
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge_deeplab_resnet(outs, base_channels, H, W)
