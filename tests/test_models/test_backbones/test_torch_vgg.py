import torch

from tiseg.models.backbones import TorchVGG


def judge(outs, C, H, W):
    assert len(outs) == 6
    assert outs[0].shape == (2, C * 2**0, H // 1, W // 1)
    assert outs[1].shape == (2, C * 2**1, H // 2, W // 2)
    assert outs[2].shape == (2, C * 2**2, H // 4, W // 4)
    assert outs[3].shape == (2, C * 2**3, H // 8, W // 8)
    assert outs[4].shape == (2, C * 2**3, H // 16, W // 16)
    assert outs[5].shape == (2, C * 2**3, H // 32, W // 32)


def test_vgg():
    base_channels = 64
    N, C, H, W = (2, 3, 224, 224)
    temp = torch.randn((N, C, H, W))
    model = TorchVGG(
        model_name='vgg16_bn',
        out_indices=(0, 1, 2, 3, 4, 5),
        pretrained=False)
    # Test model weights initilize without pretrain.
    model.init_weights()
    outs = model(temp)

    judge(outs, base_channels, H, W)
