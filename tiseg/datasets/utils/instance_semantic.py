import numpy as np
from skimage import morphology


def re_instance(instance_map):
    """convert sparse instance ids to continual instance ids for instance
    map."""
    instance_ids = list(np.unique(instance_map))
    new_instance_map = np.zeros_like(instance_map, dtype=np.int32)
    for id, instance_id in enumerate(instance_ids):
        # ignore background id
        if instance_id == 0:
            continue
        new_instance_map[instance_map == instance_id] = id + 1

    return new_instance_map


def convert_instance_to_semantic(instance_map, with_edge=True):
    """Convert instance mask to semantic mask.

    Args:
        instances (numpy.ndarray): The mask contains each instances with
            different label value.
        with_edge (bool): Convertion with edge class label.

    Returns:
        mask (numpy.ndarray): mask contains two or three classes label
            (background, nuclei)
    """
    mask = np.zeros_like(instance_map, dtype=np.uint8)
    instance_ids = list(np.unique(instance_map))
    for instance_id in instance_ids:
        single_instance_map = (instance_map == instance_id).astype(np.uint8)
        if with_edge:
            boundary = morphology.dilation(single_instance_map) & (
                ~morphology.erosion(single_instance_map))
            mask += single_instance_map
            mask[boundary > 0] = 2
        else:
            mask += single_instance_map

    return mask
