import torch

from tiseg.models.utils import UNetDecoderLayer, UNetNeckLayer


def test_unet_component():
    # Test neck layer
    temp = torch.randn((1, 12, 50, 50))
    model = UNetNeckLayer(12, 24, 4)

    out = model(temp)
    assert out.shape == (1, 24, 50, 50)

    # Test decode layer with skip connection
    temp = torch.randn((1, 12, 27, 27))
    skip = torch.randn((1, 14, 50, 50))

    model = UNetDecoderLayer(14, 32, 3, in_channels=12)

    out = model(skip, temp)
    assert out.shape == (1, 32, 50, 50)

    # Test decode layer without skip connection
    skip = torch.randn((1, 14, 50, 50))

    model = UNetDecoderLayer(14, 32, 3)

    out = model(skip)
    assert out.shape == (1, 32, 50, 50)
