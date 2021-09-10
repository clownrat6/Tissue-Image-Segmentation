import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import (binary_erosion, binary_fill_holes,
                                      distance_transform_edt)
from skimage import measure, morphology
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from ..builder import PIPELINES
from ..utils import (angle_to_vector, calculate_centerpoint,
                     calculate_gradient, vector_to_label)


@PIPELINES.register_module()
class CDNetLabelMake(object):
    """Label construction for CDNet."""

    def __init__(self, input_level='semantic_with_edge', num_angle_types=8):
        self.input_level = input_level
        self.num_angle_types = num_angle_types

    def __call__(self, results):
        assert self.input_level in ['semantic_with_edge']
        if self.input_level == 'semantic_with_edge':
            semantic_map = results['gt_semantic_map']

            # Check if the input level is "semantic_with_edge"
            # "semantic_with_edge" means the semantic map has three classes
            # (background, nuclei_inside, nuclei_edge)
            results['gt_semantic_map_with_edge'] = semantic_map
            results['gt_semantic_map'] = (semantic_map == 1).astype(np.uint8)

            semantic_map_edge = results['gt_semantic_map_with_edge']
            instance_map = measure.label(
                semantic_map_edge == 1, connectivity=1)
            # Redundant dilation will cause instance connection.
            results['gt_instance_map'] = instance_map
        else:
            raise NotImplementedError

        # point map calculation & gradient map calculation
        point_map, gradient_map, instance_map_dilation = (
            self.calculate_point_map(instance_map))

        # direction map calculation
        direction_map = self.calculate_direction_map(instance_map_dilation,
                                                     gradient_map)

        results['gt_point_map'] = point_map
        results['gt_direction_map'] = direction_map
        results['seg_fields'].append('gt_semantic_map_with_edge')
        results['seg_fields'].append('gt_direction_map')
        results['seg_fields'].append('gt_point_map')

        return results

    def calculate_direction_map(self, instance_map, gradient_map):
        # Prepare for gradient map & direction map calculation
        # instance_map = morphology.dilation(instance_map, morphology.disk(1))
        # continue angle calculation
        angle_map = np.degrees(
            np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
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
        instance_map_dilation = np.zeros((H, W), dtype=np.float32)

        markers_unique = np.unique(instance_map)
        markers_len = len(np.unique(instance_map)) - 1

        # Calculate for each instance
        for k in markers_unique[1:]:
            single_instance_map = (instance_map == k).astype(np.uint8)

            center = calculate_centerpoint(single_instance_map, H, W)
            # Count each center to judge if some instances don't get center
            assert single_instance_map[center[0], center[1]] > 0
            point_map[center[0], center[1]] = 1

            # Prepare for gradient map & direction map calculation
            single_instance_map_dilation = morphology.dilation(
                single_instance_map, morphology.disk(1))
            instance_map_dilation += single_instance_map_dilation

            # Calculate distance from points of instance to instance center.
            distance_to_center_instance = self.calculate_distance_to_center(
                single_instance_map_dilation, center)
            distance_to_center_map += distance_to_center_instance

            # Calculate gradient of (to center) distance
            gradient_map_instance = self.calculate_gradient(
                single_instance_map_dilation, distance_to_center_instance)
            gradient_map[(single_instance_map_dilation != 0), :] = 0
            gradient_map += gradient_map_instance
        assert int(point_map.sum()) == markers_len

        # Use gaussian filter to process center point map
        point_map_gaussian = gaussian_filter(
            point_map * 255, sigma=2, order=0).astype(np.float32)

        return point_map_gaussian, gradient_map, instance_map_dilation

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


@PIPELINES.register_module()
class EdgeMapCalculation(object):
    """Edge Class Calculation.

    Only support two-class semantic map now.

    Arg:
        radius (int): Morphological Operations hyper parameters. Default: 1
        edge_map_key (str): Semantic Map with edge class.
            Default: gt_semantic_map_with_edge
    """

    def __init__(self,
                 already_edge=True,
                 radius=1,
                 edge_map_key='gt_semantic_map_with_edge'):
        self.already_edge = already_edge
        self.radius = radius
        self.edge_map_key = edge_map_key

    def __call__(self, results):
        label = results['gt_semantic_map']

        # Check if the input level is "semantic_with_edge"
        # "semantic_with_edge" means the semantic map has three classes
        # (background, nuclei_inside, nuclei_edge)
        if len(np.unique(label)) == 3:
            # semantic map can't contain edge class
            results[self.edge_map_key] = label
            results['gt_semantic_map'] = (label == 1).astype(np.uint8)
            results['seg_fields'].append(self.edge_map_key)
            return results

        # Input without edge class
        assert len(np.unique(label)) == 2, 'Only support binary label now.'
        bounds = morphology.dilation(label) & (
            ~morphology.erosion(label, morphology.disk(self.radius)))
        edge_map = label.copy()

        # Assign edge pixels
        edge_map[bounds > 0] = 2
        results[self.edge_map_key] = edge_map
        results['seg_fields'].append(self.edge_map_key)

        return results


@PIPELINES.register_module()
class InstanceMapCalculation(object):
    """Calculate instances according to semantic map.

    Only support two-class semantic map now.

    Args:
        remove_small_object (bool): Whether to remove small object.
            Default: True
        object_small_size (int): The minimal size of object to remove.
            Default: 10
        radius (int): Morphological Operations hyper parameters. Default: 1
        instance_map_key (str): Instance Map converted from Semantic Map
            storage key. Default: 'gt_instance_map'
    """

    def __init__(self,
                 remove_small_object=False,
                 object_small_size=10,
                 radius=1,
                 instance_map_key='gt_instance_map'):
        self.remove_small_object = remove_small_object
        self.object_small_size = object_small_size
        self.radius = radius
        self.instance_map_key = instance_map_key

    def __call__(self, results):
        label = results['gt_semantic_map']
        assert len(np.unique(label)) <= 2, 'Only support binary label now.'
        if self.remove_small_object:
            instance_label = self.process((label == 1).astype(np.uint8))
        else:
            instance_label = measure.label((label == 1).astype(np.uint8))

        # instantiation
        results[self.instance_map_key] = instance_label
        results['seg_fields'].append(self.instance_map_key)
        return results

    def process(self, seg_map):
        # Using erosion and distance thresh operation to split connected
        # instances.
        dist = measure.label(seg_map)
        dist = self.gen_inst_dst_map(dist)
        marker = np.copy(dist)
        marker[marker <= 125] = 0
        marker[marker > 125] = 1
        marker = binary_fill_holes(marker)
        marker = binary_erosion(marker, iterations=1)
        marker = measure.label(marker)

        # Remove small instances
        marker = remove_small_objects(marker, min_size=self.object_small_size)
        seg_map = watershed(-dist, marker, mask=seg_map)
        seg_map = remove_small_objects(
            seg_map, min_size=self.object_small_size)

        return seg_map

    def gen_inst_dst_map(self, seg_map):
        shape = seg_map.shape[:2]
        nuc_list = list(np.unique(seg_map))
        nuc_list.remove(0)
        canvas = np.zeros(shape, dtype=np.uint8)
        for nuc_id in nuc_list:
            nuc_map = np.copy(seg_map == nuc_id)
            nuc_dst = distance_transform_edt(nuc_map)
            nuc_dst = 255 * (nuc_dst / np.amax(nuc_dst))
            canvas += nuc_dst.astype('uint8')

        return canvas


@PIPELINES.register_module()
class PointMapCalculation(object):
    """Calculate Point Map and Gradient Map for every instance.

    Args:
        radius (int): Morphological Operations hyper parameters. Default: 1
        point_map_key (str): Point Map storage key. Default: 'gt_point_map'
        gradient_map_key (str): Gradient Map storage key.
            Default: 'gt_gradient_map'
    """

    def __init__(self,
                 radius=1,
                 point_map_key='gt_point_map',
                 gradient_map_key='gt_gradient_map'):
        self.radius = radius
        self.point_map_key = point_map_key
        self.gradient_map_key = gradient_map_key

    def calculate_distance_to_center(self, single_instance_map, center):
        H, W = single_instance_map.shape[:2]
        # Calculate distance (to center) map for single instance
        single_instance_map = morphology.dilation(single_instance_map,
                                                  morphology.disk(self.radius))
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

    def __call__(self, results):
        seg_instance_map = results['gt_instance_map']
        H, W = seg_instance_map.shape[:2]
        # distance_center_map: The min distance between center and point
        distance_to_center_map = np.zeros((H, W), dtype=np.float32)
        gradient_map = np.zeros((H, W, 2), dtype=np.float32)
        point_map = np.zeros((H, W), dtype=np.float)

        mask = seg_instance_map
        markers_unique = np.unique(seg_instance_map)
        markers_len = len(np.unique(seg_instance_map)) - 1

        for k in markers_unique[1:]:
            single_instance_map = (mask == k).astype(np.uint8)

            center = calculate_centerpoint(single_instance_map, H, W)
            # Count each center to judge if some instances don't get center
            assert single_instance_map[center[0], center[1]] > 0
            point_map[center[0], center[1]] = 1

            distance_to_center_instance = self.calculate_distance_to_center(
                single_instance_map, center)
            distance_to_center_map += distance_to_center_instance

            gradient_map_instance = self.calculate_gradient(
                single_instance_map, distance_to_center_instance)
            gradient_map[(single_instance_map != 0), :] = 0
            gradient_map += gradient_map_instance
        assert int(point_map.sum()) == markers_len

        # Use gaussian filter to process center point map
        point_map_gaussian = gaussian_filter(
            point_map * 255, sigma=2, order=0).astype(np.float32)

        results[self.point_map_key] = point_map_gaussian
        results[self.gradient_map_key] = gradient_map
        results['seg_fields'].append(self.point_map_key)
        results['seg_fields'].append(self.gradient_map_key)

        return results


@PIPELINES.register_module()
class DirectionMapCalculation(object):
    """Calculate Direction Map & Angle Map according to gradient map.

    Direction Map divide Angle Map into multiple classes.

    Args:
        num_angle_types (int): Divide angle to multiple classes. Default: 8
        angle_map_key (str): Angle Map storage key. Default: 'gt_angle_map'
        direction_map_key (str): Direction Map storage key.
            Default: 'gt_direction_map'
    """

    def __init__(self,
                 num_angle_types=8,
                 angle_map_key='gt_angle_map',
                 direction_map_key='gt_direction_map'):
        self.num_angle_types = num_angle_types
        self.angle_map_key = angle_map_key
        self.direction_map_key = direction_map_key

    def __call__(self, results):
        instance_map = results['gt_instance_map']
        gradient_map = results['gt_gradient_map']
        # TODO: Refactor direction map calculation
        # continue angle calculation
        angle_map = np.degrees(
            np.arctan2(gradient_map[:, :, 0], gradient_map[:, :, 1]))
        vector_map = angle_to_vector(angle_map, self.num_angle_types)
        # angle type judgement
        direction_map = vector_to_label(vector_map, self.num_angle_types)

        direction_map[instance_map == 0] = -1
        direction_map = direction_map + 1
        # set median value (maybe no need)
        # direction_map = median(direction_map,
        #                        morphology.disk(1, dtype=np.int64))
        results[self.angle_map_key] = angle_map.astype(np.float32)
        results[self.direction_map_key] = direction_map.astype(np.int64)
        results['seg_fields'].append(self.angle_map_key)
        results['seg_fields'].append(self.direction_map_key)

        return results
