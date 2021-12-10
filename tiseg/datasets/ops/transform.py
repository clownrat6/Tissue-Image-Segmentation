import random

import cv2
import mmcv
import numpy as np


class ColorJitter(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, img):
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img


class Resize(object):

    def __init__(self, min_size, max_size, resize_mode):
        self.min_size = min_size
        self.max_size = max_size
        self.resize_mode = resize_mode

    def __call__(self, img, segs):
        if self.resize_mode == 'fix':
            img = cv2.resize(img, (self.min_size, self.min_size))
            for i, seg in enumerate(segs):
                segs[i] = cv2.resize(
                    seg, (self.min_size, self.min_size),
                    interpolation=cv2.INTER_NEAREST)
        elif self.resize_mode == 'scale':
            h, w = img.shape[:2]
            min_len = min(h, w)
            scale_f = self.min_size / min_len
            scale_h, scale_w = h * scale_f, w * scale_f
            img = cv2.resize(img, (scale_h, scale_w))
            for i, seg in enumerate(segs):
                segs[i] = cv2.resize(
                    seg, (scale_h, scale_w), interpolation=cv2.INTER_NEAREST)

        return img, segs


class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, list[str], optional): The flipping direction. Options
            are 'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=None, direction='horizontal'):
        if prob is not None:
            assert prob >= 0 and prob <= 1
        else:
            prob = 0
        self.prob = prob

        if isinstance(direction, list):
            assert sum([
                x in ['horizontal', 'vertical', 'diagonal'] for x in direction
            ]) == len(direction)
        else:
            assert direction in ['horizontal', 'vertical', 'diagonal']
            direction = [direction]
        self.direction = direction

    def __call__(self, img, segs):
        flip = True if np.random.rand() < self.prob else False
        # random select from direction list.
        select_index = np.random.randint(0, len(self.direction))
        flip_direction = self.direction[select_index]
        if flip:
            # flip image
            img = mmcv.imflip(img, direction=flip_direction)

            new_segs = []
            # flip segs
            for seg in segs:
                seg = mmcv.imflip(seg, direction=flip_direction)
                new_segs.append(seg)
            segs = new_segs

        return img, segs
