import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from .ops import (ColorJitter, DirectionLabelMake, RandomFlip, Resize,
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
        self.min_size = process_cfg['min_size']
        self.max_size = process_cfg['max_size']
        self.resize_mode = process_cfg['resize_mode']
        # self.size_div = process_cfg['size_div']
        self.with_dir = process_cfg['with_dir']
        self.edge_id = process_cfg['edge_id']

        self.color_jitter = ColorJitter()
        self.flipper = RandomFlip(prob=0.5)
        self.resizer = Resize(self.min_size, self.max_size, self.resize_mode)
        self.dir_label_maker = DirectionLabelMake(edge_id=self.edge_id)

    def __call__(self, data_info):
        data_info = copy.deepcopy(data_info)

        img = read_image(data_info['file_name'])
        sem_seg = read_image(data_info['sem_file_name'])
        inst_seg = read_image(data_info['inst_file_name'])

        if not self.test_mode:
            h, w = img.shape[:2]
            data_info['raw_h'] = h
            data_info['raw_w'] = w
            assert img.shape[:2] == sem_seg.shape[:2]

            if self.if_flip:
                img, segs = self.flipper(img, [sem_seg, inst_seg])
                sem_seg = segs[0]
                inst_seg = segs[1]

            img, segs = self.resizer(img, [sem_seg, inst_seg])
            sem_seg = segs[0]
            inst_seg = segs[1]

            img = self.color_jitter(img)

        h, w = img.shape[:2]
        data_info['input_h'] = h
        data_info['input_w'] = w

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

        if self.with_dir:
            res = self.dir_label_maker(sem_seg, inst_seg)
            point_reg = res['gt_point_map']
            dir_seg = res['gt_direction_map']
            ret['label']['point_gt'] = format_reg(point_reg)
            ret['label']['dir_gt'] = format_seg(dir_seg)

        return ret
