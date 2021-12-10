import numpy as np
import torch
import torch.nn as nn

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads.fast_unet_head import UNetHead
from ..losses import GeneralizedDiceLoss, miou, tiou
from .fast_base import FastBaseSegmentor


@SEGMENTORS.register_module()
class UNetSegmentor(FastBaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(UNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(
            in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4])
        self.head = UNetHead(
            num_classes=self.num_classes,
            in_dims=(64, 128, 256, 512, 512),
            stage_dims=[64, 128, 256, 512, 512],
            dropout_rate=0.1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            align_corners=False)

    def calculate(self, img):
        img_feats = self.backbone(img)
        mask_logit = self.head(img_feats)

        return mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if self.training:
            mask_logit = self.calculate(data['img'])
            assert label is not None
            mask_label = label['gt_semantic_map']
            loss = dict()
            mask_logit = resize(input=mask_logit, size=mask_label.shape[2:])
            mask_label = mask_label.squeeze(1)
            mask_loss = self._mask_loss(mask_logit, mask_label)
            loss.update(mask_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(
                mask_logit, mask_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            seg_logit = self.inference(data['img'], metas[0], True)
            seg_pred = seg_logit.argmax(dim=1)
            # Extract inside class
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred

    def _mask_loss(self, mask_logit, mask_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = GeneralizedDiceLoss(
            num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(
            mask_ce_loss_calculator(mask_logit, mask_label))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_label)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)
        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_label,
                                      self.num_classes)

        # metric calculate (the edge id is set `self.num_classes - 1` in
        # default)
        mask_pred = torch.argmax(
            mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        mask_pred[mask_pred == (self.num_classes - 1)] = 0
        mask_target = mask_label.cpu().numpy().astype(np.uint8)
        mask_target[mask_target == (self.num_classes - 1)] = 0

        N = mask_pred.shape[0]
        wrap_dict['aji'] = 0.
        for i in range(N):
            aji_single_image = aggregated_jaccard_index(
                mask_pred[i], mask_target[i])
            wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # distributed environment requires cuda tensor
        wrap_dict['aji'] /= N
        wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
