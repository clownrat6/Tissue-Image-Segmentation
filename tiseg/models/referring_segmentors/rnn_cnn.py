import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import add_prefix, resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class RnnCnn(BaseSegmentor):
    """RNN CNN Referring Expression Segmentors.

    RNN-CNN architecture from `LSTM-CNN` -
    'https://arxiv.org/pdf/1603.06180.pdf'
    """

    def __init__(self,
                 txt_backbone,
                 img_backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 txt_pretrained=None,
                 img_pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if txt_pretrained is not None:
            txt_backbone.pretrained = txt_pretrained
        if img_pretrained is not None:
            img_backbone.pretrained = img_pretrained

        self.txt_backbone = builder.build_backbone(txt_backbone)
        self.img_backbone = builder.build_backbone(img_backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head`` and extract arg: align_corners.

        The explanation of arg: align_corners:
        * zh: 'https://zhuanlan.zhihu.com/p/87572724'
        * en: 'https://discuss.pytorch.org/t/what-we-should-use-align-corners
        -false/22663/8'
        """
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``

        auxiliary_head (dict | list[dict]): There may be more than one
        auxiliary head.
        """
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img, txt):
        """Extract features from images and texts."""
        img_feats = self.img_backbone(img)
        txt_feats = self.txt_backbone(txt)
        feats = (img_feats, txt_feats)
        if self.with_neck:
            feats = self.neck(feats)
        return feats

    def _decode_head_forward_train(self, feats, metas, gt_instance_map):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(feats, metas,
                                                     gt_instance_map,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, feats, metas):
        """Run forward function for decode head in inference."""
        seg_logits = self.decode_head.forward_test(feats, metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, feats, metas, gt_instance_map):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(feats, metas,
                                                  gt_instance_map,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                feats, metas, gt_instance_map, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_simple(self, img, txt, metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        feats = self.extract_feat(img, txt)
        out = self._decode_head_forward_test(feats, metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, txt, metas, gt_instance_map):
        """Forward function for training.

        Args:
            img (Tensor): Input images (N, C, H, W).
            txt (Tensor): Input texts (N, L, C).
            metas (list[dict]): List of data info dict where each
                dict has: xxx.
                For details on the values of these keys see
                `tres/datasets/pipelines/formatting.py:Collect`.
            gt_instance_map (Tensor): Referring expression segmentation
                ground truth mask.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(img, txt)

        losses = dict()

        loss_decode = self._decode_head_forward_train(feats, metas,
                                                      gt_instance_map)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                feats, metas, gt_instance_map)
            losses.update(loss_aux)

        return losses

    def forward_test(self, img, txt, metas, gt_instance_map, **kwargs):
        """
        Args:
            img (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape
                [N, C, H, W], which contains all images in the batch.
            txtx (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape
                [N, L, C], which contains all texts in the batch.
            metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                infos of all data in a batch.
        """

        num_augs = len(img)
        if num_augs != len(txt):
            raise ValueError(f'num of augmentations ({len(img)}) != '
                             f'num of texts ({len(txt)})')
        if num_augs != len(metas):
            raise ValueError(f'num of augmentations ({len(img)}) != '
                             f'num of image meta ({len(metas)})')

        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for meta in metas:
            ori_shapes = [_['img_info']['ori_shape'] for _ in meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_info']['img_shape'] for _ in meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['img_info']['pad_shape'] for _ in meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        return self._test(img, txt, metas, **kwargs)

    def _test(self, img, txt, metas, rescale=True):
        """Simple test or Augmentation test."""

        # aug_test rescale all img back to ori_shape for now
        assert rescale or len(img) == 1, ('When multi-scale test, the rescale',
                                          'must be set to True.')

        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(img[0], txt[0], metas[0], rescale)
        for i in range(1, len(img)):
            cur_seg_logit = self.inference(img[i], txt[i], metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(img)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def inference(self, img, txt, meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            txt (Tensor): The input text of shape (N, L, 1).
            meta (dict): data info dict where each dict has: xxx.
                For details on the values of these keys see
                `tres/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = meta[0]['img_info']['ori_shape']
        assert all(_['img_info']['ori_shape'] == ori_shape for _ in meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, txt, meta, rescale)
        else:
            seg_logit = self.whole_inference(img, txt, meta, rescale)

        # Softmax may be unnecessary
        # output = F.softmax(seg_logit, dim=1)
        output = seg_logit

        # Recover from flip
        flip = meta[0]['img_info']['flip']
        if flip:
            flip_direction = meta[0]['img_info']['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            # flip on H dimension
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            # flip on W dimension
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def slide_inference(self, img, txt, meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_simple(crop_img, txt, meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=meta[0]['img_info']['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, txt, meta, rescale):
        """Inference with full image."""

        seg_logit = self.forward_simple(img, txt, meta)
        if rescale:
            size = meta[0]['img_info']['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit
