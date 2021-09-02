import torch

from ...datasets.utils import label_to_vector


def circshift(matrix, direction, shift_vertical, shift_horizontal):
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
    H, W, C = matrix.shape
    matrix_new = torch.zeros_like(matrix)

    for k in range(C):
        temp_matrix = matrix[:, :, k]
        if (direction == 1):
            # upper left
            temp_matrix = torch.vstack(
                (temp_matrix[shift_vertical:, :],
                 torch.zeros_like(temp_matrix[:shift_vertical, :])))
            temp_matrix = torch.hstack(
                (temp_matrix[:, shift_horizontal:],
                 torch.zeros_like(temp_matrix[:, :shift_horizontal])))
        elif (direction == 2):
            # upper right
            temp_matrix = torch.vstack(
                (temp_matrix[shift_vertical:, :],
                 torch.zeros_like(temp_matrix[:shift_vertical, :])))
            temp_matrix = torch.hstack(
                (torch.zeros_like(temp_matrix[:, (W - shift_horizontal):]),
                 temp_matrix[:, :(W - shift_horizontal)]))
        elif (direction == 3):
            # lower left
            temp_matrix = torch.vstack(
                (torch.zeros_like(temp_matrix[(H - shift_vertical):, :]),
                 temp_matrix[:(H - shift_vertical), :]))
            temp_matrix = torch.hstack(
                (temp_matrix[:, shift_horizontal:],
                 torch.zeros_like(temp_matrix[:, :shift_horizontal])))
        elif (direction == 4):
            # lower right
            temp_matrix = torch.vstack(
                (torch.zeros_like(temp_matrix[(H - shift_vertical):, :]),
                 temp_matrix[:(H - shift_vertical), :]))
            temp_matrix = torch.hstack(
                (torch.zeros_like(temp_matrix[:, (W - shift_horizontal):]),
                 temp_matrix[:, :(W - shift_horizontal)]))
        matrix_new[:, :, k] = temp_matrix

    return matrix_new


# TODO: regularize variable name and add doc string
def generate_direction_differential_map(direction_map, direction_classes=9):
    # label_to_vector requires NxHxW (torch.Tensor) or HxW (numpy.ndarry)
    vector_map = label_to_vector(direction_map, direction_classes)
    # Only support batch size == 1
    # Nx2xHxW (2: vertical and horizontal)
    direction_map = direction_map[0]
    vector_map = vector_map[0].permute(1, 2, 0).detach()
    vector_anchor = vector_map

    H, W = vector_anchor.shape[0], vector_anchor.shape[1]
    # Cosine Similarity Map
    cos_sim_map = torch.zeros((H, W), dtype=torch.float32)

    feature_list = []
    # Only support 8 direction now
    if direction_classes - 1 == 8:
        feature1 = circshift(vector_anchor, 1, 1, 1)
        feature2 = circshift(vector_anchor, 1, 1, 0)
        feature3 = circshift(vector_anchor, 2, 1, 1)
        feature4 = circshift(vector_anchor, 3, 0, 1)
        feature6 = circshift(vector_anchor, 4, 0, 1)
        feature7 = circshift(vector_anchor, 3, 1, 1)
        feature8 = circshift(vector_anchor, 3, 1, 0)
        feature9 = circshift(vector_anchor, 4, 1, 1)

        feature_list.append(feature1)
        feature_list.append(feature2)
        feature_list.append(feature3)
        feature_list.append(feature4)
        feature_list.append(feature6)
        feature_list.append(feature7)
        feature_list.append(feature8)
        feature_list.append(feature9)

    cos_sim_map_single_direction = torch.zeros((H, W, direction_classes - 1),
                                               dtype=torch.float32)
    for k, feature_item in enumerate(feature_list):
        numerator = (
            vector_anchor[:, :, 0] * feature_item[:, :, 0] +
            vector_anchor[:, :, 1] * feature_item[:, :, 1])
        denominator = (
            torch.sqrt(
                pow(vector_anchor[:, :, 0], 2) +
                pow(vector_anchor[:, :, 1], 2)) *
            torch.sqrt(
                pow(feature_item[:, :, 0], 2) + pow(feature_item[:, :, 1], 2))
            + 0.000001)
        cos_sim_map_single_direction[:, :, k] = numerator / denominator

    cos_sim_map, cos_sim_indices = torch.min(
        cos_sim_map_single_direction, dim=2)

    cos_sim_map[direction_map == 0] = 1

    cos_sim_map = (1 - torch.round(cos_sim_map))
    cos_sim_map_max = torch.max(cos_sim_map)
    cos_sim_map_min = torch.min(cos_sim_map)
    cos_sim_map_normal = (cos_sim_map - cos_sim_map_min) / (
        cos_sim_map_max - cos_sim_map_min)

    return cos_sim_map_normal
