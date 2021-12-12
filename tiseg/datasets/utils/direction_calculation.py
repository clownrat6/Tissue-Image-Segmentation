import numpy as np
import torch

label_to_vector_mapping = {
    4: [[-1, -1], [-1, 1], [1, 1], [1, -1]],
    5: [[0, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]],
    8: [[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]],
    9: [[0, 0], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0],
        [1, -1]],
    16: [[0, -2], [-1, -2], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
         [-1, 2], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2],
         [1, -2]],
    17: [[0, 0], [0, -2], [-1, -2], [-2, -2], [-2, -1], [-2, 0], [-2, 1],
         [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0], [2, -1],
         [2, -2], [1, -2]],
    32: [
        [0, -4],
        [-1, -4],
        [-2, -4],
        [-3, -4],
        [-4, -4],
        [-4, -3],
        [-4, -2],
        [-4, -1],
        [-4, 0],
        [-4, 1],
        [-4, 2],
        [-4, 3],
        [-4, 4],
        [-3, 4],
        [-2, 4],
        [-1, 4],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4],
        [4, 4],
        [4, 3],
        [4, 2],
        [4, 1],
        [4, 0],
        [4, -1],
        [4, -2],
        [4, -3],
        [4, -4],
        [3, -4],
        [2, -4],
        [1, -4],
    ]
}


# TODO: Add doc string and comments
def align_angle(angle_map, num_classes=8):

    assert isinstance(angle_map, np.ndarray)

    step = 360 / num_classes
    new_angle_map = np.zeros(angle_map.shape, dtype=np.float)
    angle_index_map = np.zeros(angle_map.shape, dtype=np.int)
    mask = (angle_map <= (-180 + step / 2)) | (angle_map > (180 - step / 2))
    new_angle_map[mask] = -180
    angle_index_map[mask] = 0

    for i in range(1, num_classes):
        middle = -180 + step * i
        mask = (angle_map > (middle - step / 2)) & (
            angle_map <= (middle + step / 2))
        new_angle_map[mask] = middle
        angle_index_map[mask] = i

    return new_angle_map, angle_index_map


def angle_to_vector(angle_map, num_classes=8):
    assert isinstance(angle_map, np.ndarray)

    vector_map = np.zeros((*angle_map.shape, 2), dtype=np.float)
    deg2rad = np.deg2rad

    if num_classes is not None:
        angle_map, _ = align_angle(angle_map, num_classes=num_classes)

    angle_map = deg2rad(angle_map)

    vector_map[..., 0] = np.sin(angle_map)
    vector_map[..., 1] = np.cos(angle_map)

    return vector_map


def angle_to_direction_label(angle_map,
                             seg_label_map=None,
                             num_classes=8,
                             extra_ignore_mask=None):
    assert isinstance(angle_map, np.ndarray)
    assert isinstance(seg_label_map, np.ndarray) or seg_label_map is None

    _, label_map = align_angle(angle_map, num_classes=num_classes)

    if seg_label_map is None:
        ignore_mask = np.zeros(angle_map.shape, dtype=np.bool)
    else:
        ignore_mask = seg_label_map == -1

    if extra_ignore_mask is not None:
        ignore_mask = ignore_mask | extra_ignore_mask
    label_map[ignore_mask] = -1

    return label_map


def vector_to_label(vector_map, num_classes=8):

    assert isinstance(vector_map, np.ndarray)

    rad2deg = np.rad2deg

    angle_map = np.arctan2(vector_map[..., 0], vector_map[..., 1])
    angle_map = rad2deg(angle_map)

    return angle_to_direction_label(angle_map, num_classes=num_classes)


def label_to_vector(dir_map, num_classes=8):

    assert isinstance(dir_map, torch.Tensor)

    mapping = label_to_vector_mapping[num_classes]
    offset_h = torch.zeros_like(dir_map)
    offset_w = torch.zeros_like(dir_map)

    for idx, (hdir, wdir) in enumerate(mapping):
        mask = dir_map == idx
        offset_h[mask] = hdir
        offset_w[mask] = wdir

    # vertical, horizontal direction concat
    vector_map = torch.stack([offset_h, offset_w], dim=-1)
    # NHWC -> NCHW
    vector_map = vector_map.permute(0, 3, 1, 2).to(dir_map.device)

    return vector_map
