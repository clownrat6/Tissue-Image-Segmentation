import torch

from tiseg.models.decode_heads import NucleiCDHead


def test_cd_head_with_extra_stage():
    # Test ResNet
    H, W = (224, 224)
    temp_list = [
        torch.randn((1, 16, H // 4, W // 4)),
        torch.randn((1, 32, H // 8, W // 8)),
        torch.randn((1, 64, H // 16, W // 16)),
        torch.randn((1, 128, H // 32, W // 32))
    ]
    model = NucleiCDHead(
        stage_channels=[16, 32, 64, 128],
        extra_stage_channels=256,
        stage_convs=[3, 3, 3, 3],
        extra_stage_convs=3,
        in_channels=[16, 32, 64, 128],
        in_index=[0, 1, 2, 3])

    out = model(temp_list)
    assert out[0].shape == (1, 3, 56, 56)
    assert out[1].shape == (1, 9, 56, 56)
    assert out[2].shape == (1, 1, 56, 56)

    # Test Deeplab-ResNet
    temp_list = [
        torch.randn((1, 16, H // 4, W // 4)),
        torch.randn((1, 32, H // 8, W // 8)),
        torch.randn((1, 64, H // 8, W // 8)),
        torch.randn((1, 128, H // 8, W // 8))
    ]
    out = model(temp_list)
    assert out[0].shape == (1, 3, 56, 56)
    assert out[1].shape == (1, 9, 56, 56)
    assert out[2].shape == (1, 1, 56, 56)


def test_cd_head_without_extra_stage():
    # Test ResNet
    H, W = (224, 224)
    temp_list = [
        torch.randn((1, 15, H // 4, W // 4)),
        torch.randn((1, 31, H // 8, W // 8)),
        torch.randn((1, 63, H // 16, W // 16)),
        torch.randn((1, 127, H // 32, W // 32))
    ]
    model = NucleiCDHead(
        stage_channels=[16, 32, 64, 128],
        extra_stage_channels=None,
        stage_convs=[3, 3, 3, 3],
        extra_stage_convs=None,
        in_channels=[15, 31, 63, 127],
        in_index=[0, 1, 2, 3])

    out = model(temp_list)
    assert out[0].shape == (1, 3, 56, 56)
    assert out[1].shape == (1, 9, 56, 56)
    assert out[2].shape == (1, 1, 56, 56)

    # Test Deeplab-ResNet
    temp_list = [
        torch.randn((1, 15, H // 4, W // 4)),
        torch.randn((1, 31, H // 8, W // 8)),
        torch.randn((1, 63, H // 8, W // 8)),
        torch.randn((1, 127, H // 8, W // 8))
    ]
    out = model(temp_list)
    assert out[0].shape == (1, 3, 56, 56)
    assert out[1].shape == (1, 9, 56, 56)
    assert out[2].shape == (1, 1, 56, 56)
