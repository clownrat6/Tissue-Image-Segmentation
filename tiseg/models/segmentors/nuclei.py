import torch
import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import add_prefix, resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class Nuclei(BaseSegmentor):
    """Segmentor for Nuclei Segmentation.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(Nuclei, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, metas, label):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, metas, label,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, x, metas, label):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, metas, label,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, metas, label, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _decode_head_forward_test(self, x, metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, metas, self.test_cfg)
        return seg_logits

    def forward_train(self, data, metas, label):
        """Forward function for training.

        Args:
            data (dict): Input data wrapper, inner structure:
                data = dict('img': Tensor (NxCxHxW)).
            metas (list[dict]): List of data info dict where each dict
                has: 'img_info', 'ann_info'.
            label (dict): Label wrapper, inner structure:
                label = dict('gt_semantic_map': Tensor (NxCxHxW).

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(data['img'])

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, metas, label)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, metas, label)
            losses.update(loss_aux)

        return losses

    def forward_test(self, data, metas, label, **kwargs):
        """
        Args:
            data (List[dict]): test-time augmentations data wrapper list and
                inner structure:
                    [dict('img': Tensor (NxCxHxW)), ...]
            metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
            label: The placeholder to compat with forward_train
        """
        imgs = [single_data['img'] for single_data in data]
        assert label == [], (
            'There is no need keeping "label" key when evaluation.')
        for var, name in [(imgs, 'imgs'), (metas, 'metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(data)
        if num_augs != len(metas):
            raise ValueError(f'num of augmentations ({len(data)}) != '
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

        if num_augs == 1:
            return self.simple_test(imgs[0], metas[0], **kwargs)
        else:
            return self.aug_test(imgs, metas, **kwargs)

    # TODO refactor
    def slide_inference(self, img, meta, rescale):
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
                crop_seg_logit = self.encode_decode(crop_img, meta)
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

    # TODO: refactor code stryle
    def slide_inference_plus(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = (self.test_cfg.crop_size[0] -
                        self.test_cfg.stride[0]) * 2

        N, C, H, W = img.shape

        input = img

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (
                window_size - overlap_size)
            tmp = torch.zeros((N, C, pad_h, W)).to(img.device)
            input = torch.cat((input, tmp), dim=2)

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (
                window_size - overlap_size)
            tmp = torch.zeros((N, C, H + pad_h, pad_w)).to(img.device)
            input = torch.cat((input, tmp), dim=3)

        _, C1, H1, W1 = input.size()

        output = torch.zeros((input.size(0), 3, H1, W1)).to(img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (
                i + window_size -
                overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch
                output_patch = self.encode_decode(input_var, meta)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (
                    j + window_size -
                    overlap_size // 2 if j + window_size < W1 else W1)
                output[:, :, ind1_s:ind1_e,
                       ind2_s:ind2_e] = output_patch[:, :,
                                                     ind1_s - i:ind1_e - i,
                                                     ind2_s - j:ind2_e - j]

        output = output[:, :, :H, :W]
        if rescale:
            output = resize(
                output,
                size=meta[0]['img_info']['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return output

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, meta)
        if rescale:
            size = meta[0]['img_info']['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict where each dict has: 'img_info',
                'ann_info'
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = meta[0]['img_info']['ori_shape']
        assert all(_['img_info']['ori_shape'] == ori_shape for _ in meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, meta, rescale)
        else:
            seg_logit = self.whole_inference(img, meta, rescale)
        output = seg_logit
        # output = F.softmax(seg_logit, dim=1)
        flip = meta[0]['img_info']['flip']
        rotate = meta[0]['img_info']['rotate']
        # reverse tta must have reverse order of origin tta
        if rotate:
            rotate_degree = meta[0]['img_info']['rotate_degree']
            assert rotate_degree in [90, 180, 270]
            # torch.rot90 has reverse direction of mmcv.imrotate
            # TODO: recover rotate output (Need to conside the flip operation.)
            if rotate_degree == 90:
                output = output.rot90(dims=(2, 3))
            elif rotate_degree == 180:
                output = output.rot90(k=2, dims=(2, 3))
            elif rotate_degree == 270:
                output = output.rot90(k=3, dims=(2, 3))
        if flip:
            flip_direction = meta[0]['img_info']['flip_direction']
            assert flip_direction in ['horizontal', 'vertical', 'diagonal']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
            elif flip_direction == 'diagonal':
                output = output.flip(dims=(2, 3))

        return output

    def simple_test(self, img, meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        # Extract inside class
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        # Extract inside class
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
