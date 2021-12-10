import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology

from ..utils import (angle_to_vector, calculate_centerpoint,
                     calculate_gradient, vector_to_label)


class DirectionLabelMake(object):
    """build direction label & point label for any dataset."""

    def __init__(self, edge_id, re_edge=True, num_angle_types=8):
        # If input with edge, re_edge can be set to False.
        # However, in order to generate better boundary, we will re-generate
        # edge.
        self.edge_id = edge_id
        self.re_edge = re_edge
        self.num_angle_types = num_angle_types

    def __call__(self, sem_map, inst_map):
        results = {}

        if self.re_edge:
            sem_map_w_edge = np.zeros_like(sem_map)
            id_list = list(np.unique(sem_map))
            for id in id_list:
                if id == self.edge_id or id == 0:
                    continue
                id_mask = sem_map == id
                bound = morphology.dilation(
                    id_mask,
                    selem=morphology.selem.disk(1)) & (~morphology.erosion(
                        id_mask, selem=morphology.selem.disk(1)))
                sem_map_w_edge[id_mask > 0] = id
                sem_map_w_edge[bound > 0] = self.edge_id

            sem_map_in = sem_map_w_edge.copy()
            sem_map_in[sem_map_in == self.edge_id] = 0
            results['gt_sem_map'] = sem_map_w_edge
        else:
            sem_map_in = sem_map.copy()
            sem_map_in[sem_map_in == self.edge_id] = 0
            sem_map_w_edge = sem_map

        results['gt_sem_map_in'] = sem_map_in
        results['gt_sem_map_w_edge'] = sem_map_w_edge

        instance_map = inst_map

        # point map calculation & gradient map calculation
        point_map, gradient_map = self.calculate_point_map(instance_map)

        # direction map calculation
        direction_map = self.calculate_direction_map(instance_map,
                                                     gradient_map)

        results['gt_point_map'] = point_map
        results['gt_direction_map'] = direction_map

        return results

    def calculate_direction_map(self, instance_map, gradient_map):
        # Prepare for gradient map & direction map calculation
        # continue angle calculation
        angle_map = np.degrees(
            np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        angle_map[instance_map == 0] = 0
        vector_map = angle_to_vector(angle_map, self.num_angle_types)
        # angle type judgement
        direction_map = vector_to_label(vector_map, self.num_angle_types)

        direction_map[instance_map == 0] = -1
        direction_map = direction_map + 1

        return direction_map

    def calculate_point_map(self, instance_map):
        H, W = instance_map.shape[:2]
        # distance_center_map: The min distance between center and point
        distance_to_center_map = np.zeros((H, W), dtype=np.float32)
        gradient_map = np.zeros((H, W, 2), dtype=np.float32)
        point_map = np.zeros((H, W), dtype=np.float32)

        # remove background
        markers_unique = list(np.unique(instance_map))
        markers_len = len(markers_unique) - 1

        # Calculate for each instance
        for k in markers_unique:
            if k == 0:
                continue
            single_instance_map = (instance_map == k).astype(np.uint8)

            center = calculate_centerpoint(single_instance_map, H, W)
            # Count each center to judge if some instances don't get center
            assert single_instance_map[center[0], center[1]] > 0
            point_map[center[0], center[1]] = 1

            # Calculate distance from points of instance to instance center.
            distance_to_center_instance = self.calculate_distance_to_center(
                single_instance_map, center)
            distance_to_center_map += distance_to_center_instance

            # Calculate gradient of (to center) distance
            gradient_map_instance = self.calculate_gradient(
                single_instance_map, distance_to_center_instance)
            gradient_map[(single_instance_map != 0), :] = 0
            gradient_map += gradient_map_instance
        assert int(point_map.sum()) == markers_len

        # Use gaussian filter to process center point map
        point_map_gaussian = gaussian_filter(
            point_map * 255, sigma=2, order=0).astype(np.float32)

        return point_map_gaussian, gradient_map

    def calculate_distance_to_center(self, single_instance_map, center):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        point_map_instance = np.zeros((H, W), dtype=np.uint8)
        point_map_instance[center[0], center[1]] = 1
        distance_to_center = distance_transform_edt(1 - point_map_instance)
        # Only calculate distance (to center) in distance region
        distance_to_center = distance_to_center * single_instance_map
        distance_to_center_instance = (
            1 - distance_to_center /
            (distance_to_center.max() + 0.0000001)) * single_instance_map

        return distance_to_center_instance

    def calculate_gradient(self, single_instance_map,
                           distance_to_center_instance):
        H, W = single_instance_map.shape[:2]
        gradient_map_instance = np.zeros((H, W, 2))
        gradient_map_instance = calculate_gradient(
            distance_to_center_instance, ksize=11)
        gradient_map_instance[(single_instance_map == 0), :] = 0
        return gradient_map_instance
