import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from skimage import measure, morphology

from ..builder import PIPELINES
from ..utils import (angle_to_vector, calculate_centerpoint,
                     calculate_gradient, vector_to_label)


@PIPELINES.register_module()
class CDNetLabelMake(object):
    """Label construction for CDNet."""

    def __init__(self,
                 input_level='semantic_with_edge',
                 re_edge=True,
                 num_angle_types=8):
        # If input with edge, re_edge can be set to False.
        # However, in order to generate better boundary, we will re-generate
        # edge.
        if re_edge:
            assert input_level == 'semantic_with_edge'
        self.re_edge = re_edge
        self.input_level = input_level
        self.num_angle_types = num_angle_types

    def __call__(self, results):
        assert self.input_level in ['instance', 'semantic_with_edge']
        if self.input_level == 'semantic_with_edge':
            raw_semantic_map = results['gt_semantic_map']

            # Check if the input level is "semantic_with_edge"
            # "semantic_with_edge" means the semantic map has three classes
            # (background, nuclei_inside, nuclei_edge)
            if self.re_edge:
                semantic_map_inside = (raw_semantic_map == 1).astype(np.uint8)
                bound = morphology.dilation(
                    semantic_map_inside,
                    selem=morphology.selem.disk(1)) & (~morphology.erosion(
                        semantic_map_inside, selem=morphology.selem.disk(1)))
                # fuse boundary & inside
                semantic_map_with_edge = np.zeros_like(raw_semantic_map)
                semantic_map_with_edge[semantic_map_inside > 0] = 1
                semantic_map_with_edge[bound > 0] = 2
            else:
                semantic_map_inside = (raw_semantic_map == 1).astype(np.uint8)
                semantic_map_with_edge = raw_semantic_map

            results['gt_semantic_map_inside'] = semantic_map_inside
            results['gt_semantic_map_with_edge'] = semantic_map_with_edge

            instance_map = measure.label(
                semantic_map_inside == 1, connectivity=1)

            # XXX: If re_edge, we need to dilate two pixels during
            # model-agnostic postprocess.
            if self.re_edge:
                # re_edge will remove a pixel length of nuclei inside and raw
                # semantic map has already remove a pixel length of nuclei
                # inside, so we need to dilate 2 pixel length.
                instance_map = morphology.dilation(
                    instance_map, selem=morphology.selem.disk(2))
            else:
                instance_map = morphology.dilation(
                    instance_map, selem=morphology.selem.disk(1))

            results['gt_instance_map'] = instance_map
            results['gt_semantic_map'] = (instance_map > 0).astype(np.uint8)
        elif self.input_level == 'instance':
            # build semantic map from instance map
            instance_map = results['gt_semantic_map']
            semantic_map_with_edge = np.zeros_like(
                instance_map, dtype=np.uint8)
            instance_id_list = list(np.unique(instance_map))
            # remove background id
            instance_id_list.remove(0)
            for instance_id in instance_id_list:
                single_instance_map = (instance_map == instance_id).astype(
                    np.uint8)

                bound = morphology.dilation(single_instance_map) & (
                    ~morphology.erosion(single_instance_map))

                semantic_map_with_edge[single_instance_map > 0] = 1
                semantic_map_with_edge[bound > 0] = 2

            results['gt_semantic_map_inside'] = (
                semantic_map_with_edge == 1).astype(np.uint8)
            results['gt_semantic_map_with_edge'] = semantic_map_with_edge
            results['gt_instance_map'] = instance_map
            results['gt_semantic_map'] = (instance_map > 0).astype(np.uint8)
        else:
            raise NotImplementedError

        # point map calculation & gradient map calculation
        point_map, gradient_map = self.calculate_point_map(instance_map)

        # direction map calculation
        direction_map = self.calculate_direction_map(instance_map,
                                                     gradient_map)

        results['gt_point_map'] = point_map
        results['gt_direction_map'] = direction_map

        additional_key_list = [
            'gt_semantic_map', 'gt_semantic_map_with_edge', 'gt_direction_map',
            'gt_point_map'
        ]
        for additional_key in additional_key_list:
            if additional_key not in results['seg_fields']:
                results['seg_fields'].append(additional_key)

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

        markers_unique = np.unique(instance_map)
        markers_len = len(np.unique(instance_map)) - 1

        # Calculate for each instance
        for k in markers_unique[1:]:
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


@PIPELINES.register_module()
class CityscapesLabelMake(object):
    """Label construction for every network on cityscapes dataset."""

    def __init__(self,
                 num_classes,
                 edge_label_id,
                 re_edge=True,
                 num_angle_types=8):
        # If input with edge, re_edge can be set to False.
        # However, in order to generate better boundary, we will re-generate
        # edge.
        self.num_classes = num_classes
        self.edge_label_id = edge_label_id
        self.re_edge = re_edge
        self.num_angle_types = num_angle_types

    def __call__(self, results):
        raw_semantic_map = results['gt_semantic_map']
        raw_instance_map = results['gt_instance_map']

        # remove semantic label
        raw_instance_map[raw_instance_map < 1000] = 0

        if self.re_edge:
            semantic_map_with_edge = np.zeros_like(raw_semantic_map)
            for label_id in range(1, self.num_classes):
                if label_id == self.edge_label_id:
                    continue
                single_class_inside = (raw_semantic_map == label_id).astype(
                    np.uint8)
                bound = morphology.dilation(
                    single_class_inside,
                    selem=morphology.selem.disk(1)) & (~morphology.erosion(
                        single_class_inside, selem=morphology.selem.disk(1)))
                semantic_map_with_edge[single_class_inside > 0] = label_id
                semantic_map_with_edge[bound > 0] = self.edge_label_id

            semantic_map_inside = semantic_map_with_edge.copy()
            semantic_map_inside[semantic_map_inside == self.edge_label_id] = 0
        else:
            semantic_map_inside = raw_semantic_map.copy()
            semantic_map_inside[semantic_map_inside == self.edge_label_id] = 0
            semantic_map_with_edge = raw_semantic_map

        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(semantic_map_inside)
        # plt.subplot(122)
        # plt.imshow(semantic_map_with_edge)
        # plt.savefig('4.png')
        # exit(0)
        results['gt_semantic_map_inside'] = semantic_map_inside
        results['gt_semantic_map_with_edge'] = semantic_map_with_edge

        instance_map = raw_instance_map
        # instance_map = measure.label(semantic_map_inside > 0, connectivity=1)

        # # XXX: If re_edge, we need to dilate two pixels during
        # # model-agnostic postprocess.
        # if self.re_edge:
        #     # re_edge will remove a pixel length of nuclei inside and raw
        #     # semantic map has already remove a pixel length of nuclei
        #     # inside, so we need to dilate 2 pixel length.
        #     instance_map = morphology.dilation(
        #         instance_map, selem=morphology.selem.disk(2))
        # else:
        #     instance_map = morphology.dilation(
        #         instance_map, selem=morphology.selem.disk(1))

        results['gt_instance_map'] = instance_map
        results['gt_semantic_map'] = (instance_map > 0).astype(np.uint8)

        # point map calculation & gradient map calculation
        point_map, gradient_map = self.calculate_point_map(instance_map)

        # direction map calculation
        direction_map = self.calculate_direction_map(instance_map,
                                                     gradient_map)

        results['gt_point_map'] = point_map
        results['gt_direction_map'] = direction_map

        additional_key_list = [
            'gt_semantic_map', 'gt_semantic_map_with_edge', 'gt_direction_map',
            'gt_point_map'
        ]
        for additional_key in additional_key_list:
            if additional_key not in results['seg_fields']:
                results['seg_fields'].append(additional_key)

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
        markers_unique.remove(0)
        markers_len = len(markers_unique)

        # Calculate for each instance
        for k in markers_unique:
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
