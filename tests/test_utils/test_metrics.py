import os.path as osp

import numpy as np
from PIL import Image

from tiseg.utils.evaluation.metrics import aggregated_jaccard_index


def test_aji():
    pseudo_seg_map_path = osp.join(
        osp.dirname(__file__), '../data/nucleus_label_map.png')
    pseudo_seg_map = Image.open(pseudo_seg_map_path)
    pseudo_semantic_map = np.array(pseudo_seg_map)[:, :, 0]
    pseudo_semantic_map[pseudo_semantic_map == 255] = 1

    result = aggregated_jaccard_index(pseudo_semantic_map, pseudo_semantic_map)

    assert result == 1.0
