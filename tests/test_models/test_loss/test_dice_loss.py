import torch
import torch.nn.functional as F

from tiseg.models.losses import DiceLoss, GeneralizedDiceLoss


def test_dice_loss():
    N, C, H, W = (4, 3, 14, 14)

    criterion = DiceLoss(C)

    logit = F.softmax(torch.randn((N, C, H, W)), dim=1)
    target = torch.randint(0, C, (N, H, W))

    loss_value = criterion(logit, target)

    print(loss_value)

    criterion = GeneralizedDiceLoss(C)

    loss_value = criterion(logit, target)

    print(loss_value)

    exit(0)

    assert loss_value >= 0 and loss_value <= 1


def test_generalized_dice_loss():
    N, C, H, W = (4, 3, 14, 14)

    criterion = GeneralizedDiceLoss(C)

    logit = F.softmax(torch.randn((N, C, H, W)), dim=1)
    target = torch.randint(0, C, (N, H, W))

    loss_value = criterion(logit, target)

    assert loss_value >= 0 and loss_value <= 1
