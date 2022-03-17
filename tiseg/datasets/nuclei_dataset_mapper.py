import copy
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from .ops import class_dict, BoundLabelMake, format_img, format_info, format_seg


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

    def __init__(self, test_mode, *, processes):
        self.test_mode = test_mode

        self.augments = []
        for process in processes:
            class_name = process.pop('type')
            augment = class_dict[class_name](**process)
            self.augments.append(augment)

        self.basic_label_maker = BoundLabelMake(edge_id=2)

    def __call__(self, data_info):
        data_info = copy.deepcopy(data_info)

        img = read_image(data_info['file_name'])
        sem_seg = read_image(data_info['sem_file_name'])
        inst_seg = read_image(data_info['inst_file_name'])

        data_info['ori_hw'] = img.shape[:2]

        h, w = img.shape[:2]
        assert img.shape[:2] == sem_seg.shape[:2]

        segs = [sem_seg, inst_seg]
        for augment in self.augments:
            img, segs = augment(img, segs)

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

        if not self.test_mode:
            res = self.basic_label_maker(sem_seg, inst_seg)
            sem_seg_w_bound = res['sem_gt_w_bound']
            ret['label']['sem_gt_w_bound'] = format_seg(sem_seg_w_bound)

        return ret
