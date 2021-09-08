import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tiseg.datasets.pipelines import (CDNetLabelMake, DirectionMapCalculation,
                                      EdgeMapCalculation,
                                      InstanceMapCalculation,
                                      PointMapCalculation)
from tiseg.models.utils import generate_direction_differential_map


def test_split_label_calculation():
    pseudo_seg_map_path = osp.join(
        osp.dirname(__file__), '../data/example_nuclei_label.png')
    pseudo_seg_map = Image.open(pseudo_seg_map_path)
    pseudo_semantic_map = np.array(pseudo_seg_map)
    pseudo_semantic_map[pseudo_semantic_map == 2] = 1

    results_raw = {}
    results_raw['gt_semantic_map'] = pseudo_semantic_map
    results_raw['seg_fields'] = ['gt_semantic_map']

    # Test InstanceLabelCalculation
    transform = InstanceMapCalculation(remove_small_object=False)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(231)
    plt.imshow(pseudo_instance_map)

    # Test InstanceLabelCalculation with remove small object
    transform = InstanceMapCalculation(
        remove_small_object=True, object_small_size=10)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(232)
    plt.imshow(pseudo_instance_map)

    # Test EdgeLabelCalculation
    transform = EdgeMapCalculation(already_edge=False)
    results_edge = transform(results_raw)
    pseudo_semantic_map_edge = results_edge['gt_semantic_map_with_edge']
    assert len(np.unique(pseudo_semantic_map_edge)) == len(
        np.unique(pseudo_semantic_map)) + 1
    plt.subplot(233)
    plt.imshow(pseudo_semantic_map_edge)

    # Test PointMapCalculation
    transform = PointMapCalculation()
    results_point = transform(results_instance)
    pseudo_point_map = results_point['gt_point_map']
    assert pseudo_point_map.dtype == np.float32
    plt.subplot(234)
    plt.imshow(pseudo_point_map)

    # Test DirectionMapCalculation
    transform = DirectionMapCalculation(num_angle_types=8)
    results_direction = transform(results_point)
    pseudo_angle_map = results_direction['gt_angle_map']
    pseudo_direction_map = results_direction['gt_direction_map']
    plt.subplot(235)
    plt.imshow(pseudo_angle_map)
    plt.subplot(236)
    plt.imshow(pseudo_direction_map)
    plt.show()


def test_general_label_calculation():
    pseudo_seg_map_path = osp.join(
        osp.dirname(__file__), '../data/example_nuclei_label.png')
    pseudo_seg_map = Image.open(pseudo_seg_map_path)
    pseudo_semantic_map = np.array(pseudo_seg_map)

    results_raw = {}
    results_raw['gt_semantic_map'] = pseudo_semantic_map
    results_raw['seg_fields'] = ['gt_semantic_map']

    transform = CDNetLabelMake()
    results = transform(results_raw)
    plt.subplot(221)
    plt.imshow(results['gt_semantic_map'])
    plt.subplot(222)
    plt.imshow(results['gt_semantic_map_with_edge'])
    plt.subplot(223)
    plt.imshow(results['gt_point_map'])
    plt.subplot(224)
    plt.imshow(results['gt_direction_map'])
    plt.show()

    # Test DirectionDifferentialMap
    inference_direction_map = torch.from_numpy(
        results['gt_direction_map']).expand(1, -1, -1)
    ddm = generate_direction_differential_map(inference_direction_map, 9)
    ddm = ddm.numpy()[0]
    plt.subplot(121)
    plt.imshow(ddm)
    plt.subplot(122)
    temp = results['gt_semantic_map_with_edge']
    canvas = np.zeros((*temp.shape, 3), dtype=np.uint8)
    canvas[temp == 2, :] = (255, 255, 2)
    canvas[ddm > 0, :] = (2, 255, 255)
    plt.imshow(canvas)
    plt.show()

    gt_direction_map = results['gt_direction_map'][100:200, 100:200]
    gt_semantic_map_with_edge = results['gt_semantic_map_with_edge'][100:200,
                                                                     100:200]
    gt_direction_map = cv2.resize(
        gt_direction_map, (320, 320), interpolation=cv2.INTER_NEAREST)
    gt_semantic_map_with_edge = cv2.resize(
        gt_semantic_map_with_edge, (320, 320), interpolation=cv2.INTER_NEAREST)

    plt.subplot(221)
    plt.imshow(gt_semantic_map_with_edge == 2)
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(gt_direction_map)
    plt.axis('off')
    plt.subplot(223)
    inference_direction_map = torch.from_numpy(gt_direction_map).expand(
        1, -1, -1)
    ddm = generate_direction_differential_map(inference_direction_map, 9)
    ddm = ddm.numpy()[0]
    plt.imshow(ddm)
    plt.axis('off')
    plt.subplot(224)
    canvas = np.zeros((*gt_direction_map.shape, 3), dtype=np.uint8)
    canvas[gt_semantic_map_with_edge == 2, :] = (255, 0, 0)
    canvas[gt_direction_map > 0, :] = (0, 0, 255)
    canvas[ddm > 0, :] = (0, 255, 0)
    plt.imshow(canvas)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
