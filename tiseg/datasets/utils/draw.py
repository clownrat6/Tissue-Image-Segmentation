import random

import numpy as np


def colorize_seg_map(seg_map, palette=None):
    """using random rgb color to colorize segmentation map."""
    colorful_seg_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    id_list = list(np.unique(seg_map))

    if palette is None:
        palette = {}
        for id in id_list:
            color = [random.random() * 255 for i in range(3)]
            palette[id] = color

    for id in id_list:
        # ignore background
        if id == 0:
            continue
        colorful_seg_map[seg_map == id, :] = palette[id]

    return colorful_seg_map
