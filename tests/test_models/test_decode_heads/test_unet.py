import torch

from tiseg.models.decode_heads import NucleiUNetHead


def test_nuclei_unet():
    # Test ResNet
    H, W = (224, 224)
    temp_list = [
        torch.randn((1, 16, H // 4, W // 4)),
        torch.randn((1, 32, H // 8, W // 8)),
        torch.randn((1, 64, H // 16, W // 16)),
        torch.randn((1, 128, H // 32, W // 32))
    ]
    model = NucleiUNetHead(
        in_channels=[16, 32, 64, 128],
        num_classes=3,
        in_index=[0, 1, 2, 3],
        stage_channels=[16, 32, 64, 128],
        stage_convs=[3, 3, 3, 3])
    out = model(temp_list)
    assert out.shape == (1, 3, 56, 56)

    label = {'gt_semantic_map_with_edge': torch.randint(0, 2, (1, 1, H, W))}
    loss = model.forward_train(temp_list, None, label, None)
    assert isinstance(loss, dict)
    assert ('mask_ce_loss'
            in loss) and ('aji' in loss) and ('mask_dice'
                                              in loss) and ('mask_iou' in loss)
