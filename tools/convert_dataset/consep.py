import argparse
import os
import os.path as osp
import shutil

import numpy as np
from PIL import Image
from rich.progress import track
from scipy.io import loadmat
from skimage import morphology


def convert_mat_to_array(mat, array_key='inst_map', save_path=None):
    """Convert matlab format array file to numpy array file."""
    if isinstance(mat, str):
        mat = loadmat(mat)

    mat = mat[array_key]

    if save_path is not None:
        pass

    return mat


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
    H, W = instance_map.shape
    semantic_map = np.zeros([H, W], dtype=np.uint8)
    instance_id_list = list(np.unique(instance_map))
    for id in instance_id_list:
        if id == 0:
            continue
        single_instance_map = instance_map == id
        if with_edge:
            boundary = morphology.dilation(single_instance_map) & (
                ~morphology.erosion(single_instance_map))
            semantic_map += single_instance_map
            semantic_map[boundary > 0] = 2
        else:
            semantic_map += single_instance_map

    return semantic_map


def convert_each_cohort(raw_path, new_path):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    raw_image_folder = osp.join(raw_path, 'Images')
    raw_label_folder = osp.join(raw_path, 'Labels')

    item_list = [osp.splitext(x)[0] for x in os.listdir(raw_image_folder)]

    for item in track(item_list):
        image_path = osp.join(raw_image_folder, item + '.png')
        label_path = osp.join(raw_label_folder, item + '.mat')

        image = np.array(Image.open(image_path))
        instance_label = convert_mat_to_array(label_path)
        semantic_label = convert_instance_to_semantic(
            instance_label, with_edge=False)
        semantic_label_edge = convert_instance_to_semantic(
            instance_label, with_edge=True)

        assert image.shape[:2] == instance_label.shape

        # Save image
        dst_image_path = osp.join(new_path, item + '.png')
        shutil.copy(image_path, dst_image_path)

        # Save instance level label
        dst_instance_label_path = osp.join(new_path, item + '_instance.png')
        instance_label_png = Image.fromarray(semantic_label_edge)
        instance_label_png.save(dst_instance_label_path)

        # Save semantic level label
        dst_semantic_edge_label_path = osp.join(
            new_path, item + '_semantic_with_edge.png')
        semantic_label_edge_png = Image.fromarray(semantic_label_edge)
        semantic_label_edge_png.save(dst_semantic_edge_label_path)

        dst_semantic_label_path = osp.join(new_path, item + '_semantic.png')
        semantic_label_png = Image.fromarray(semantic_label)
        semantic_label_png.save(dst_semantic_label_path)

    return item_list


def parse_args():
    parser = argparse.ArgumentParser('Convert consep dataset.')
    parser.add_argument('root_path', help='dataset root path.')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    train_raw_path = osp.join(root_path, 'CoNSeP', 'Train')
    test_raw_path = osp.join(root_path, 'CoNSeP', 'Test')

    train_new_path = osp.join(root_path, 'train')
    test_new_path = osp.join(root_path, 'test')

    # make train cohort dataset
    item_list = convert_each_cohort(train_raw_path, train_new_path)
    with open(osp.join(root_path, 'train.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in item_list]

    # make test cohort dataset
    item_list = convert_each_cohort(test_raw_path, test_new_path)
    with open(osp.join(root_path, 'test.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in item_list]


if __name__ == '__main__':
    main()
