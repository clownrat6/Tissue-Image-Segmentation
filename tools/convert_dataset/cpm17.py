import argparse
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage import morphology

from tiseg.datasets.utils import colorize_seg_map


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
    instance_id_list.remove(0)
    for id in instance_id_list:
        single_instance_map = instance_map == id
        if with_edge:
            boundary = morphology.dilation(single_instance_map) & (
                ~morphology.erosion(single_instance_map))
            semantic_map += single_instance_map
            semantic_map[boundary > 0] = 2
        else:
            semantic_map += single_instance_map

    return semantic_map


def pillow_save(save_path, array, palette=None):
    """storage image array by using pillow."""
    image = Image.fromarray(array.astype(np.uint8))
    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)
    image.save(save_path)


def crop_patches(image, crop_size, crop_stride):
    """crop image into several patches according to the crop size & slide
    stride."""
    h_crop = w_crop = crop_size
    h_stride = w_stride = crop_stride

    assert image.ndim >= 2

    h_img, w_img = image.shape[:2]

    image_patch_list = []

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = image[y1:y2, x1:x2]

            image_patch_list.append(crop_img)

    return image_patch_list


def parse_single_item(item, raw_image_folder, raw_label_folder, new_path,
                      crop_size, crop_stride):
    """meta process of single item data."""

    image_path = osp.join(raw_image_folder, item + '.png')
    label_path = osp.join(raw_label_folder, item + '.mat')

    # image & label extraction
    image = np.array(Image.open(image_path))
    instance_label = convert_mat_to_array(label_path)
    semantic_label = convert_instance_to_semantic(
        instance_label, with_edge=False)
    semantic_label_edge = convert_instance_to_semantic(
        instance_label, with_edge=True)

    # split map into patches
    if crop_size is not None:
        crop_stride = crop_stride
        image_patches = crop_patches(image, crop_size, crop_stride)
        instance_patches = crop_patches(instance_label, crop_size, crop_stride)
        semantic_patches = crop_patches(semantic_label, crop_size, crop_stride)
        semantic_edge_patches = crop_patches(semantic_label_edge, crop_size,
                                             crop_stride)

        assert len(image_patches) == len(instance_patches) == len(
            semantic_patches) == len(semantic_edge_patches)

        item_len = len(image_patches)
        # record patch item name
        sub_item_list = [
            f'{item}_{i}_c{crop_size}_s{crop_stride}' for i in range(item_len)
        ]
    else:
        image_patches = [image]
        instance_patches = [instance_label]
        semantic_patches = [semantic_label]
        semantic_edge_patches = [semantic_label_edge]
        # record patch item name
        sub_item_list = [item]

    # patch storage
    patch_batches = zip(image_patches, instance_patches, semantic_patches,
                        semantic_edge_patches)
    for patch, sub_item in zip(patch_batches, sub_item_list):
        # jump when exists
        if osp.exists(osp.join(new_path, sub_item + '.png')):
            continue
        # save image
        pillow_save(osp.join(new_path, sub_item + '.png'), patch[0])
        # save instance level label
        np.save(osp.join(new_path, sub_item + '_instance.npy'), patch[1])
        pillow_save(
            osp.join(new_path, sub_item + '_instance_colorized.png'),
            colorize_seg_map(patch[1]))
        # save semantic level label
        palette = np.zeros((2, 3), dtype=np.uint8)
        palette[0, :] = (0, 0, 0)
        palette[1, :] = (255, 255, 2)
        pillow_save(
            osp.join(new_path, sub_item + '_semantic.png'), patch[2], palette)
        palette = np.zeros((3, 3), dtype=np.uint8)
        palette[0, :] = (0, 0, 0)
        palette[1, :] = (255, 0, 0)
        palette[2, :] = (0, 255, 0)
        pillow_save(
            osp.join(new_path, sub_item + '_semantic_with_edge.png'), patch[3],
            palette)

    return {item: sub_item_list}


def convert_cohort(raw_image_folder,
                   raw_label_folder,
                   new_path,
                   item_list,
                   crop_size=None,
                   crop_stride=None):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    fix_kwargs = {
        'raw_image_folder': raw_image_folder,
        'raw_label_folder': raw_label_folder,
        'new_path': new_path,
        'crop_size': crop_size,
        'crop_stride': crop_stride
    }

    meta_process = partial(parse_single_item, **fix_kwargs)

    real_item_dict = {}
    results = mmcv.track_parallel_progress(meta_process, item_list, 4)
    [real_item_dict.update(result) for result in results]

    return real_item_dict


def parse_args():
    parser = argparse.ArgumentParser('Convert cpm17 dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument(
        '-c',
        '--crop-size',
        type=int,
        help='the crop size of fix crop in dataset convertion operation')
    parser.add_argument(
        '-s',
        '--crop-stride',
        type=int,
        help='the crop slide stride of fix crop')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    crop_size = args.crop_size
    crop_stride = args.crop_stride

    flag1 = (crop_size is not None) and (crop_stride is not None)
    flag2 = (crop_size is None) and (crop_stride is None)
    assert flag1 or flag2, (
        '--crop-size and --crop-stride only valid when both of them are set')

    if crop_size is not None:
        train_part_name = f'train_c{crop_size}_s{crop_stride}'
        val_part_name = 'val'
        test_part_name = 'test'
    else:
        train_part_name = 'train'
        val_part_name = 'val'
        test_part_name = 'test'

    train_raw_path = osp.join(root_path, 'CPM17', 'train')
    test_raw_path = osp.join(root_path, 'CPM17', 'test')

    train_new_path = osp.join(root_path, train_part_name)
    test_new_path = osp.join(root_path, test_part_name)

    # make train cohort dataset
    train_image_folder = osp.join(train_raw_path, 'Images')
    train_label_folder = osp.join(train_raw_path, 'Labels')

    # make test cohort dataset
    test_image_folder = osp.join(test_raw_path, 'Images')
    test_label_folder = osp.join(test_raw_path, 'Labels')

    # record convertion item
    full_train_item_list = [
        x.rstrip('.png') for x in os.listdir(train_image_folder) if '.png' in x
    ]
    full_test_item_list = [
        x.rstrip('.png') for x in os.listdir(test_image_folder) if '.png' in x
    ]

    # convertion main loop
    real_train_item_dict = convert_cohort(train_image_folder,
                                          train_label_folder, train_new_path,
                                          full_train_item_list, crop_size,
                                          crop_stride)
    _ = convert_cohort(test_image_folder, test_label_folder, test_new_path,
                       full_test_item_list, None, None)

    train_item_list = [
        x.rstrip('.png') for x in os.listdir(train_image_folder) if '.png' in x
    ]
    val_item_list = None
    test_item_list = [
        x.rstrip('.png') for x in os.listdir(test_image_folder) if '.png' in x
    ]

    real_train_item_list = []
    [
        real_train_item_list.extend(real_train_item_dict[x])
        for x in train_item_list
    ]
    real_val_item_list = val_item_list
    real_test_item_list = test_item_list

    with open(osp.join(root_path, f'{train_part_name}.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in real_train_item_list]
    with open(osp.join(root_path, f'{test_part_name}.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in real_test_item_list]

    if real_val_item_list is not None:
        with open(osp.join(root_path, f'{val_part_name}.txt'), 'w') as fp:
            [fp.write(item + '\n') for item in real_val_item_list]


if __name__ == '__main__':
    main()
