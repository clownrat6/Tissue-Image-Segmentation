import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tiseg.datasets.pipelines.transforms import (EdgeLabelCalculation,
                                                 InstanceLabelCalculation)


def test_label_calculation():
    pseudo_seg_map = osp.join(
        osp.dirname(__file__), '../data/example_nucleus.png')
    pseudo_seg_map = Image.open(pseudo_seg_map)
    pseudo_semantic_map = np.array(pseudo_seg_map)[:, :, 0]
    pseudo_semantic_map[pseudo_semantic_map == 255] = 1

    results_raw = {}
    results_raw['gt_semantic_map'] = pseudo_semantic_map

    # Test InstanceLabelCalculation
    transform = InstanceLabelCalculation(remove_small_object=False)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(131)
    plt.imshow(pseudo_instance_map)

    # Test InstanceLabelCalculation with remove small object
    transform = InstanceLabelCalculation(
        remove_small_object=True, object_small_size=10)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(132)
    plt.imshow(pseudo_instance_map)

    # Test EdgeLabelCalculation
    transform = EdgeLabelCalculation()
    results_edge = transform(results_raw)
    pseudo_semantic_map_edge = results_edge['gt_semantic_map_edge']
    assert len(np.unique(pseudo_semantic_map_edge)) == len(
        np.unique(pseudo_semantic_map)) + 1
    plt.subplot(133)
    plt.imshow(pseudo_semantic_map_edge)

    plt.show()
