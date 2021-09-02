import torch

from ...datasets.utils import label_to_vector


def circshift(matrix, shift_vertical, shift_horizontal):
    """Cyclic shift of matrix.

    direction:
    1. Upper left;
    2. Upper right;
    3. Lower left;
    4. Lower right;

    Args:
        matrix (torch.Tensor): The matrix to cyclic shift.
        direction (int): The direction selection argument.
        shift_vertical (int): The shift distance of vertical direction.
        shift_horizontal (int): The shift distance of horizontal direction.
    """
    # upper left
    moved_matrix = torch.roll(
        matrix, shifts=(shift_vertical, shift_horizontal), dims=(-2, -1))

    return moved_matrix


# TODO: regularize variable name and add doc string
def generate_direction_differential_map(direction_map, direction_classes=9):
    # label_to_vector requires NxHxW (torch.Tensor) or HxW (numpy.ndarry)
    vector_map = label_to_vector(direction_map, direction_classes)
    # Only support batch size == 1
    # Nx2xHxW (2: vertical and horizontal)
    vector_anchor = vector_map

    N, _, H, W = vector_anchor.shape
    # Cosine Similarity Map
    cos_sim_map = torch.zeros((N, H, W), dtype=torch.float32)

    feature_list = []
    # Only support 8 direction now
    if direction_classes - 1 == 8:
        # Lower
        lower = circshift(vector_anchor, 1, 0)
        # Lower Right
        lower_right = circshift(vector_anchor, 1, 1)
        # Right
        right = circshift(vector_anchor, 0, 1)
        # Upper Right
        upper_right = circshift(vector_anchor, -1, 1)
        # Upper
        upper = circshift(vector_anchor, -1, 0)
        # Upper Left
        upper_left = circshift(vector_anchor, -1, -1)
        # Left
        left = circshift(vector_anchor, 0, -1)
        # Lower Left
        lower_left = circshift(vector_anchor, 1, -1)

        feature_list.append(lower)
        feature_list.append(lower_right)
        feature_list.append(right)
        feature_list.append(upper_right)
        feature_list.append(upper)
        feature_list.append(upper_left)
        feature_list.append(left)
        feature_list.append(lower_left)

    cos_sim_map_single_direction = torch.zeros(
        (N, direction_classes - 1, H, W), dtype=torch.float32)
    for k, feature_item in enumerate(feature_list):
        numerator = (
            vector_anchor[:, 0, :, :] * feature_item[:, 0, :, :] +
            vector_anchor[:, 1, :, :] * feature_item[:, 1, :, :])
        denominator = (
            torch.sqrt(
                pow(vector_anchor[:, 0, :, :], 2) +
                pow(vector_anchor[:, 1, :, :], 2)) * torch.sqrt(
                    pow(feature_item[:, 0, :, :], 2) +
                    pow(feature_item[:, 1, :, :], 2)) + 0.000001)
        cos_sim_map_single_direction[:, k, :, :] = numerator / denominator

    cos_sim_map, cos_sim_indices = torch.min(
        cos_sim_map_single_direction, dim=1)

    cos_sim_map[direction_map == 0] = 1

    cos_sim_map = (1 - torch.round(cos_sim_map))
    cos_sim_map_max = torch.max(cos_sim_map)
    cos_sim_map_min = torch.min(cos_sim_map)
    cos_sim_map_normal = (cos_sim_map - cos_sim_map_min) / (
        cos_sim_map_max - cos_sim_map_min)

    return cos_sim_map_normal
