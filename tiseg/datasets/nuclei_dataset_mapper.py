import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from tiseg.datasets.ops.hv_map import HVLabelMake

from .ops import (Resize, ColorJitter, DirectionLabelMake, DistanceLabelMake, Identity, GenBound, RandomBlur,
                  RandomFlip, RandomElasticDeform, RandomCrop, Normalize, Pad, format_, format_img, format_info,
                  format_reg, format_seg)


def read_image(path):
    _, suffix = osp.splitext(osp.basename(path))
    if suffix == '.tif':
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif suffix == '.npy':
        img = np.load(path)
    else:
        img = Image.open(path)
        img = np.array(img)

    return img


class NucleiDatasetMapper(object):

    def __init__(self, test_mode, *, process_cfg):
        self.test_mode = test_mode

        # training argument
        self.if_resize = process_cfg.get('if_resize', False)
        self.if_flip = process_cfg.get('if_flip', False)
        self.if_jitter = process_cfg.get('if_jitter', False)
        self.if_elastic = process_cfg.get('if_elastic', False)
        self.if_blur = process_cfg.get('if_blur', False)
        self.if_crop = process_cfg.get('if_crop', False)
        self.if_pad = process_cfg.get('if_pad', False)
        self.if_norm = process_cfg.get('if_norm', False)
        self.to_center = process_cfg.get('to_center', True)
        self.num_angles = process_cfg.get('num_angles', 8)
        self.num_classes = process_cfg.get('num_classes', 2)

        self.with_dir = process_cfg.get('with_dir', False)
        self.with_dist = process_cfg.get('with_dist', False)
        self.with_hv = process_cfg.get('with_hv', False)
        self.use_distance = process_cfg.get('use_distance', False)

        self.min_size = process_cfg['min_size']
        self.max_size = process_cfg['max_size']
        self.resize_mode = process_cfg['resize_mode']
        self.edge_id = process_cfg['edge_id']

        self.resizer = Resize(self.min_size, self.max_size, self.resize_mode) if self.if_resize else Identity()
        self.color_jitter = ColorJitter() if self.if_jitter else Identity()
        self.flipper = RandomFlip(prob=0.5, direction=['horizontal']) if self.if_flip else Identity()
        self.deformer = RandomElasticDeform(prob=0.5) if self.if_elastic else Identity()
        self.bluer = RandomBlur(prob=0.5) if self.if_blur else Identity()
        self.cropper = RandomCrop((self.min_size, self.min_size)) if self.if_crop else Identity()
        self.padder = Pad(self.min_size) if self.if_pad else Identity()

        self.basic_label_maker = GenBound(edge_id=self.edge_id)
        if self.with_dir:
            self.label_maker = DirectionLabelMake(
                edge_id=self.edge_id,
                to_center=self.to_center,
                num_angles=self.num_angles,
                use_distance=self.use_distance)
        elif self.with_dist:
            self.label_maker = DistanceLabelMake(self.num_classes)
        elif self.with_hv:
            self.label_maker = HVLabelMake()

        # monuseg dataset tissue image mean & std
        nuclei_mean = [0.68861804, 0.46102882, 0.61138992]
        nuclei_std = [0.19204499, 0.20979484, 0.1658672]
        self.normalizer = Normalize(nuclei_mean, nuclei_std, self.if_norm)

    def __call__(self, data_info):
        data_info = copy.deepcopy(data_info)

        img = read_image(data_info['file_name'])
        sem_seg = read_image(data_info['sem_file_name'])
        inst_seg = read_image(data_info['inst_file_name'])

        data_info['ori_hw'] = img.shape[:2]

        if not self.test_mode:
            h, w = img.shape[:2]
            assert img.shape[:2] == sem_seg.shape[:2]

            segs = [sem_seg, inst_seg]

            # 1. Random Color
            # 2. Random Horizontal Flip
            # 3. Random Elastic Transform
            # 4. Random Crop
            img, segs = self.resizer(img, segs)
            img = self.color_jitter(img)
            img, segs = self.flipper(img, segs)
            img, segs = self.deformer(img, segs)
            img = self.bluer(img)
            img, segs = self.cropper(img, segs)
            img, segs = self.padder(img, segs)
            img = self.normalizer(img)
            sem_seg = segs[0]
            inst_seg = segs[1]
        else:
            img, _ = self.resizer(img, [])
            img = self.normalizer(img)

        h, w = img.shape[:2]
        data_info['input_hw'] = (h, w)

        img_dc = format_img(img)
        sem_dc = format_seg(sem_seg)
        inst_dc = format_seg(inst_seg)
        info_dc = format_info(data_info)

        ret = {
            'data': {
                'img': img_dc
            },
            'label': {
                'sem_gt': sem_dc,
                'inst_gt': inst_dc,
            },
            'metas': info_dc,
        }

        if not self.test_mode:
            if self.with_dir:
                res = self.label_maker(sem_seg, inst_seg)
                sem_seg_w_bound = res['sem_gt_w_bound']
                point_reg = res['point_gt']
                dir_seg = res['dir_gt']
                reg_dir_seg = res['reg_dir_gt']
                weight_map = res['loss_weight_map']
                ret['label']['sem_gt_w_bound'] = format_seg(sem_seg_w_bound)
                ret['label']['point_gt'] = format_reg(point_reg)
                ret['label']['dir_gt'] = format_seg(dir_seg)
                ret['label']['reg_dir_gt'] = format_reg(reg_dir_seg)
                ret['label']['loss_weight_map'] = format_reg(weight_map)
            elif self.with_dist:
                res = self.label_maker(sem_seg, inst_seg)
                dist_reg = res['dist_gt']
                ret['label']['dist_gt'] = format_reg(dist_reg)
            elif self.with_hv:
                res = self.label_maker(sem_seg, inst_seg)
                hv_map = res['hv_gt']
                ret['label']['hv_gt'] = format_(hv_map)

            if not self.with_dir:
                res = self.basic_label_maker(sem_seg, inst_seg)
                sem_seg_w_bound = res['sem_gt_w_bound']
                ret['label']['sem_gt_w_bound'] = format_seg(sem_seg_w_bound)

        return ret
