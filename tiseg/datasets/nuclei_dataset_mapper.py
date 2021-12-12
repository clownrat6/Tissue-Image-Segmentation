import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from .ops import (ColorJitter, DirectionLabelMake, Identity, ReEdge, RandomFlip, RandomElasticDeform, RandomCrop,
                  format_img, format_info, format_reg, format_seg)


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
        self.if_flip = process_cfg['if_flip']
        self.if_jitter = process_cfg['if_jitter']
        self.if_elastic = process_cfg['if_elastic']
        self.if_crop = process_cfg['if_crop']
        self.with_dir = process_cfg['with_dir']

        self.min_size = process_cfg['min_size']
        self.max_size = process_cfg['max_size']
        self.resize_mode = process_cfg['resize_mode']
        self.with_dir = process_cfg['with_dir']
        self.edge_id = process_cfg['edge_id']

        self.color_jitter = ColorJitter() if self.if_jitter else Identity()
        self.flipper = RandomFlip(prob=0.5) if self.if_flip else Identity()
        self.deformer = RandomElasticDeform(prob=0.8) if self.if_elastic else Identity()
        self.cropper = RandomCrop((self.min_size, self.min_size)) if self.if_crop else Identity()
        self.label_maker = DirectionLabelMake(edge_id=self.edge_id) if self.with_dir else ReEdge(edge_id=self.edge_id)

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
            img = self.color_jitter(img)
            img, segs = self.flipper(img, segs)
            img, segs = self.deformer(img, segs)
            img, segs = self.cropper(img, segs)

            sem_seg = segs[0]
            inst_seg = segs[1]

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

        if self.with_dir and not self.test_mode:
            res = self.label_maker(sem_seg, inst_seg)
            sem_seg = res['gt_sem_map']
            point_reg = res['gt_point_map']
            dir_seg = res['gt_direction_map']
            ret['label']['sem_gt'] = format_seg(sem_seg)
            ret['label']['point_gt'] = format_reg(point_reg)
            ret['label']['dir_gt'] = format_seg(dir_seg)
        else:
            res = self.label_maker(sem_seg)
            sem_seg = res['gt_sem_map']
            ret['label']['sem_gt'] = format_seg(sem_seg)

        return ret
