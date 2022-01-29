import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage import morphology

from ..utils import (angle_to_vector, calculate_centerpoint, calculate_gradient, vector_to_label)
from ...models.utils import generate_direction_differential_map


class GenBound:
    """Generate high quality boundary labels.

    The return is fixed to a three-class map (background, foreground, boundary).
    """

    def __init__(self, edge_id=2):
        self.edge_id = edge_id

    def __call__(self, sem_map, inst_map):
        """generate boundary label from instance map and pure semantic map.

        sem_map:
            0: background
            1: semantic_class 1
            2: semantic class 2
            ...

        inst_map:
            0: background
            1: instance 1
            2: instance 2
            ...

        sem_map_w_bound:
            0: background
            1: foreground
            2: boundary

        Args:
            sem_map: two-class or multi-class semantic map without edge which is
                the raw semantic map.
            inst_map: instance map with each instance id. Use inst_map = inst_id
                to extrach each instance.
        """

        sem_map_w_bound = np.zeros_like(sem_map)
        sem_map_w_bound += sem_map

        # NOTE: sem_map must match inst_map
        assert np.allclose(sem_map > 0, inst_map > 0)
        inst_id_list = list(np.unique(inst_map))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_id_mask = inst_map == inst_id
            bound = inst_id_mask & (~morphology.erosion(inst_id_mask, selem=morphology.selem.disk(2)))
            sem_map_w_bound[bound > 0] = self.edge_id

        results = {}

        # NOTE: sem_map is raw semantic map (two-class or multi-class without boundary)
        # NOTE: sem_map_w_bound is always a three-class map (background, foreground, edge)
        results['sem_gt'] = sem_map
        results['sem_gt_w_bound'] = sem_map_w_bound

        return results


class DirectionLabelMake(object):
    """build direction label & point label for any dataset."""

    def __init__(self, edge_id, to_center=True, num_angles=8, use_distance=False):
        # If input with edge, re_edge can be set to False.
        # However, in order to generate better boundary, we will re-generate
        # edge.
        self.edge_id = edge_id
        self.to_center = to_center
        self.num_angles = num_angles
        self.use_distance = use_distance

    def __call__(self, sem_map, inst_map):
        """generate boundary label & direction from instance map and pure semantic map.

        sem_map:
            0: background
            1: semantic_class 1
            2: semantic class 2
            ...

        inst_map:
            0: background
            1: instance 1
            2: instance 2
            ...

        sem_map_w_bound:
            0: background
            1: foreground
            2: boundary

        Args:
            sem_map: two-class or multi-class semantic map without edge which is
                the raw semantic map.
            inst_map: instance map with each instance id. Use inst_map = inst_id
                to extrach each instance.
        """
        results = {}

        sem_map_w_bound = np.zeros_like(sem_map)
        sem_map_w_bound += sem_map

        # NOTE: sem_map must match inst_map
        assert np.allclose(sem_map > 0, inst_map > 0)
        inst_id_list = list(np.unique(inst_map))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_id_mask = inst_map == inst_id
            bound = inst_id_mask & (~morphology.erosion(inst_id_mask, selem=morphology.selem.disk(2)))
            sem_map_w_bound[bound > 0] = self.edge_id

        # NOTE: sem_map is raw semantic map (two-class or multi-class without boundary)
        # NOTE: sem_map_w_bound is always a three-class map (background, foreground, edge)
        results['sem_gt'] = sem_map
        results['sem_gt_w_bound'] = sem_map_w_bound

        # point map calculation & gradient map calculation
        point_map, gradient_map, dist_map = self.calculate_point_map(inst_map, to_center=self.to_center)

        # direction map calculation
        dir_map = self.calculate_dir_map(inst_map, gradient_map, self.num_angles)
        reg_dir_map = self.calculate_regression_dir_map(inst_map, gradient_map)
        if self.num_angles == 8:
            weight_map = self.calculate_weight_map(dir_map, dist_map, self.num_angles)
            # weight_map = weight_map * 10.
        else:
            weight_map = np.zeros_like(dir_map)
        if self.use_distance:
            results['point_gt'] = point_map
        else:
            results['point_gt'] = dist_map
        results['dir_gt'] = dir_map
        results['reg_dir_gt'] = reg_dir_map
        results['loss_weight_map'] = weight_map

        return results

    @classmethod
    def calculate_weight_map(self, dir_map, dist_map, num_angle_types):
        # torch style api
        dd_map = generate_direction_differential_map(dir_map, num_angle_types + 1)
        dd_map = dd_map[0].numpy()
        weight_map = dd_map * (1 - dist_map)
        weight_map = morphology.dilation(dd_map, selem=morphology.selem.disk(1))
        weight_map = (weight_map.astype(np.float32) * 2 + 0.2)

        return weight_map

    @classmethod
    def calculate_dir_map(self, instance_map, gradient_map, num_angle_types):
        # Prepare for gradient map & direction map calculation
        # continue angle calculation
        angle_map = np.degrees(np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        angle_map[instance_map == 0] = 0
        vector_map = angle_to_vector(angle_map, num_angle_types)
        # angle type judgement
        dir_map = vector_to_label(vector_map, num_angle_types)

        dir_map[instance_map == 0] = -1
        dir_map = dir_map + 1

        return dir_map

    @classmethod
    def calculate_regression_dir_map(self, instance_map, gradient_map):
        angle_map = np.degrees(np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        angle_map[angle_map < 0] += 360
        angle_map[instance_map == 0] = 0

        return angle_map / 180 * np.pi

    @classmethod
    def calculate_point_map(self, instance_map, to_center=True):
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
            if to_center:
                distance_to_center_instance = self.calculate_distance_to_center(single_instance_map, center)
                distance_to_center_map += distance_to_center_instance
            else:
                distance_to_center_instance = self.calculate_distance_to_centralridge(single_instance_map)
                distance_to_center_map += distance_to_center_instance

            # Calculate gradient of (to center) distance
            gradient_map_instance = self.calculate_gradient(single_instance_map, distance_to_center_instance)
            gradient_map[(single_instance_map != 0), :] = 0
            gradient_map += gradient_map_instance
        assert int(point_map.sum()) == markers_len

        # Use gaussian filter to process center point map
        point_map_gaussian = gaussian_filter(point_map * 255, sigma=2, order=0).astype(np.float32)

        return point_map_gaussian, gradient_map, distance_to_center_map

    @classmethod
    def calculate_distance_to_center(self, single_instance_map, center):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        point_map_instance = np.zeros((H, W), dtype=np.uint8)
        point_map_instance[center[0], center[1]] = 1
        distance_to_center = distance_transform_edt(1 - point_map_instance)
        # Only calculate distance (to center) in distance region
        distance_to_center = distance_to_center * single_instance_map
        distance_to_center_instance = (1 - distance_to_center /
                                       (distance_to_center.max() + 0.0000001)) * single_instance_map

        return distance_to_center_instance

    @classmethod
    def calculate_distance_to_centralridge(self, single_instance_map):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        distance_to_centralridge = distance_transform_edt(single_instance_map)
        # Only calculate distance (to center) in distance region
        distance_to_centralridge = distance_to_centralridge * single_instance_map
        distance_to_centralridge_instance = (distance_to_centralridge /
                                             (distance_to_centralridge.max() + 0.0000001)) * single_instance_map

        return distance_to_centralridge_instance

    @classmethod
    def calculate_gradient(self, single_instance_map, distance_to_center_instance):
        H, W = single_instance_map.shape[:2]
        gradient_map_instance = np.zeros((H, W, 2))
        gradient_map_instance = calculate_gradient(distance_to_center_instance, ksize=11)
        gradient_map_instance[(single_instance_map == 0), :] = 0
        return gradient_map_instance
