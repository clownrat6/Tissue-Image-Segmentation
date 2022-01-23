import torch
import torch.nn as nn

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import UNetHead
from ..losses import mdice, tdice
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class DIST(BaseSegmentor):

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(DIST, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = UNetHead(
            num_classes=self.num_classes,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        dist_logit = self.head(bottom_feat, skip_feats)
        dist_logit = resize(input=dist_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return dist_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            dist_logit = self.calculate(data['img'])
            assert label is not None
            dist_label = label['dist_gt']
            loss = dict()
            dist_label = dist_label.squeeze(1)
            mask_loss = self._dist_loss(dist_logit, dist_label)
            loss.update(mask_loss)
            # calculate training metric
            # training_metric_dict = self._training_metric(mask_logit, mask_label)
            # loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            seg_logit = self.inference(data['img'], metas[0], True)
            # Extract inside class
            seg_logit = seg_logit.cpu().numpy()
            # unravel batch dim
            seg_logit = list(seg_logit)
            ret_list = []
            for seg in seg_pred:
                ret_list.append({'sem_pred': seg})
            return ret_list

    def _dist_loss(self, dist_logit, dist_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_mse_loss_calculator = nn.MSELoss(reduction='none')
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_mse_loss = torch.mean(mask_mse_loss_calculator(dist_logit, dist_label))
        # loss weight
        alpha = 1
        mask_loss['mask_ce_loss'] = alpha * mask_mse_loss

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
