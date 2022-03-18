import torch
import torch.nn as nn
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import UNetHead
from ..losses import BatchMultiClassDiceLoss, mdice, tdice
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class UNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(UNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = UNetHead(
            num_classes=self.num_classes + 1,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        mask_logit = self.head(bottom_feat, skip_feats)

        return mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            mask_logit = self.calculate(data['img'])
            assert label is not None
            mask_label = label['sem_gt_w_bound']
            loss = dict()
            mask_label = mask_label.squeeze(1)
            mask_loss = self._mask_loss(mask_logit, mask_label)
            loss.update(mask_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, mask_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit = self.inference(data['img'], metas[0], True)
            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            sem_pred = sem_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(sem_pred)
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        pred[pred == self.num_classes] = 0
        sem_id_list = list(np.unique(pred))
        inst_pred = np.zeros_like(pred).astype(np.int32)
        sem_pred = np.zeros_like(pred).astype(np.uint8)
        cur = 0
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            inst_sem_mask = measure.label(sem_id_mask)
            inst_sem_mask = morphology.dilation(inst_sem_mask, selem=morphology.disk(3))
            inst_sem_mask[inst_sem_mask > 0] += cur
            inst_pred[inst_sem_mask > 0] = 0
            inst_pred += inst_sem_mask
            cur += len(np.unique(inst_sem_mask))
            sem_pred[inst_sem_mask > 0] = sem_id

        return sem_pred, inst_pred

    def _mask_loss(self, mask_logit, mask_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes + 1)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(mask_ce_loss_calculator(mask_logit, mask_label))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_label)
        # loss weight
        alpha = 5
        beta = 0.5
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _training_metric(self, mask_logit, mask_label):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_label = mask_label.clone().detach()

        wrap_dict['mask_tdice'] = tdice(clean_mask_logit, clean_mask_label, self.num_classes)
        wrap_dict['mask_mdice'] = mdice(clean_mask_logit, clean_mask_label, self.num_classes)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # (the edge id is set `self.num_classes - 1` in default)
        # mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        # mask_pred[mask_pred == (self.num_classes - 1)] = 0
        # mask_target = mask_label.cpu().numpy().astype(np.uint8)
        # mask_target[mask_target == (self.num_classes - 1)] = 0

        # N = mask_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(mask_pred[i], mask_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
