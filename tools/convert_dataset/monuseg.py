import argparse
import math
import os
import os.path as osp
from functools import partial

import cv2
import mmcv
import numpy as np
from lxml import etree
from PIL import Image
from skimage import morphology

# dataset split
only_train_split_dict = {
    'train': [
        'TCGA-A7-A13E-01Z-00-DX1', 'TCGA-A7-A13F-01Z-00-DX1',
        'TCGA-AR-A1AK-01Z-00-DX1', 'TCGA-B0-5711-01Z-00-DX1',
        'TCGA-HE-7128-01Z-00-DX1', 'TCGA-HE-7129-01Z-00-DX1',
        'TCGA-18-5592-01Z-00-DX1', 'TCGA-38-6178-01Z-00-DX1',
        'TCGA-49-4488-01Z-00-DX1', 'TCGA-G9-6336-01Z-00-DX1',
        'TCGA-G9-6348-01Z-00-DX1', 'TCGA-G9-6356-01Z-00-DX1'
    ],
    'val': [
        'TCGA-AR-A1AS-01Z-00-DX1', 'TCGA-HE-7130-01Z-00-DX1',
        'TCGA-50-5931-01Z-00-DX1', 'TCGA-G9-6363-01Z-00-DX1'
    ],
    'test1': [
        'TCGA-E2-A1B5-01Z-00-DX1', 'TCGA-E2-A14V-01Z-00-DX1',
        'TCGA-B0-5710-01Z-00-DX1', 'TCGA-B0-5698-01Z-00-DX1',
        'TCGA-21-5784-01Z-00-DX1', 'TCGA-21-5786-01Z-00-DX1',
        'TCGA-CH-5767-01Z-00-DX1', 'TCGA-G9-6362-01Z-00-DX1'
    ],
    'test2': [
        'TCGA-DK-A2I6-01A-01-TS1', 'TCGA-G2-A2EK-01A-02-TSB',
        'TCGA-AY-A8YK-01A-01-TS1', 'TCGA-NH-A8F7-01A-01-TS1',
        'TCGA-KB-A93J-01A-01-TS1', 'TCGA-RD-A8N9-01A-01-TS1'
    ]
}


def extract_contours(path):
    """Extract contours (multiple vertexs) from xml file.

    Args:
        path (pathlib.PosixPath): path to the annotation file.
    Returns:
        list: list of contours with each annotation encoded
           as numpy.ndarray values.
    """
    tree = etree.parse(path)
    regions = tree.xpath('/Annotations/Annotation/Regions/Region')
    contours = []
    for region in regions:
        points = []
        for point in region.xpath('Vertices/Vertex'):
            points.append([
                math.floor(float(point.attrib['X'])),
                math.floor(float(point.attrib['Y']))
            ])

        contours.append(np.array(points, dtype=np.int32))
    return contours


def convert_contour_to_instance(contours, height, width):
    """Make the mask image from the contour annotations and image sizes.

    Args:
        contours (list): list of contours with each annotation encoded
           as numpy.ndarray values.
        height (int): The height of mask.
        width (int): The width of mask.

    Returns:
        mask (numpy.ndarray): mask contains multiple instances with different
            label value.
    """

    mask = np.zeros([height, width], dtype=np.float32)
    red, green, blue = 0, 0, 0
    index_value = 0
    for contour in contours:
        # Compress value to avoid overflow
        red = 1 + index_value / 10
        red = float(f'{red:.2f}')
        mask = cv2.drawContours(
            mask, [contour], 0, (red, green, blue), thickness=cv2.FILLED)
        index_value = index_value + 1

    return mask


def convert_instance_to_semantic(instances,
                                 height,
                                 width,
                                 num_contours,
                                 with_edge=True):
    """Convert instance mask to semantic mask.

    Args:
        instances (numpy.ndarray): The mask contains each instances with
            different label value.
        height (int): The height of mask.
        width (int): The width of mask.
        with_edge (bool): Convertion with edge class label.

    Returns:
        mask (numpy.ndarray): mask contains two or three classes label
            (background, nuclei)
    """
    mask = np.zeros([height, width], dtype=np.uint8)
    for i in range(num_contours):
        value = 1 + i / 10
        value = float(f'{value:.2f}')
        single_instance_map = instances == value
        if with_edge:
            boundary = morphology.dilation(single_instance_map) & (
                ~morphology.erosion(single_instance_map))
            mask += single_instance_map
            mask[boundary > 0] = 2
        else:
            mask += single_instance_map

    return mask


def pillow_save(save_path, array):
    """storage image array by using pillow."""
    array = Image.fromarray(array)
    array.save(save_path)


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

    image_path = osp.join(raw_image_folder, item + '.tif')
    label_path = osp.join(raw_label_folder, item + '.xml')

    # image & label extraction
    image = cv2.imread(image_path)
    H, W = image.shape[:2]
    contours = extract_contours(label_path)
    instance_label = convert_contour_to_instance(contours, H, W)
    semantic_label = convert_instance_to_semantic(
        instance_label, H, W, len(contours), with_edge=False)
    semantic_label_edge = convert_instance_to_semantic(
        instance_label, H, W, len(contours), with_edge=True)

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
        if osp.exists(osp.join(new_path, sub_item + '.tif')):
            continue
        # save image
        cv2.imwrite(osp.join(new_path, sub_item + '.tif'), patch[0])
        # save instance level label
        np.save(osp.join(new_path, sub_item + '_instance.npy'), patch[1])
        # save semantic level label
        pillow_save(osp.join(new_path, sub_item + '_semantic.png'), patch[2])
        pillow_save(
            osp.join(new_path, sub_item + '_semantic_with_edge.png'), patch[3])

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
    parser = argparse.ArgumentParser('Convert monuseg dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument('split', help='split mode selection.')
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
    split = args.split
    crop_size = args.crop_size
    crop_stride = args.crop_stride

    assert split in ['official', 'only-train']

    flag1 = (crop_size is not None) and (crop_stride is not None)
    flag2 = (crop_size is None) and (crop_stride is None)
    assert flag1 or flag2, (
        '--crop-size and --crop-stride only valid when both of them are set')

    train_raw_path = osp.join(root_path, 'MoNuSeg',
                              'MoNuSeg 2018 Training Data',
                              'MoNuSeg 2018 Training Data')
    test_raw_path = osp.join(root_path, 'MoNuSeg', 'MoNuSegTestData',
                             'MoNuSegTestData')

    if crop_size is not None:
        train_part_name = f'train_c{crop_size}_s{crop_stride}'
        val_part_name = 'val'
        test_part_name = 'test'
    else:
        train_part_name = 'train'
        val_part_name = 'val'
        test_part_name = 'test'

    # storage path of convertion dataset
    train_new_path = osp.join(root_path, train_part_name)
    test_new_path = osp.join(root_path, test_part_name)

    # make train cohort dataset
    train_image_folder = osp.join(train_raw_path, 'Tissue Images')
    train_label_folder = osp.join(train_raw_path, 'Annotations')

    # make test cohort dataset
    test_image_folder = test_raw_path
    test_label_folder = test_raw_path

    # record convertion item
    full_train_item_list = [
        x.rstrip('.tif') for x in os.listdir(train_image_folder) if '.tif' in x
    ]
    full_test_item_list = [
        x.rstrip('.tif') for x in os.listdir(test_image_folder) if '.tif' in x
    ]

    # convertion main loop
    real_train_item_dict = convert_cohort(train_image_folder,
                                          train_label_folder, train_new_path,
                                          full_train_item_list, crop_size,
                                          crop_stride)
    _ = convert_cohort(test_image_folder, test_label_folder, test_new_path,
                       full_test_item_list, None, None)

    if split == 'official':
        train_item_list = [
            x.rstrip('.tif') for x in os.listdir(train_image_folder)
            if '.tif' in x
        ]
        val_item_list = None
        test_item_list = [
            x.rstrip('.tif') for x in os.listdir(test_image_folder)
            if '.tif' in x
        ]
    elif split == 'only-train':
        train_item_list = only_train_split_dict[
            'train'] + only_train_split_dict['val']
        val_item_list = None
        test_item_list = only_train_split_dict[
            'test1'] + only_train_split_dict['test2']
    elif split == 'only-train_with_val':
        train_item_list = only_train_split_dict['train']
        val_item_list = only_train_split_dict['val']
        test_item_list = only_train_split_dict[
            'test1'] + only_train_split_dict['test2']

    real_train_item_list = []
    [
        real_train_item_list.extend(real_train_item_dict[x])
        for x in train_item_list
    ]
    real_test_item_list = test_item_list

    with open(osp.join(root_path, f'{split}_{train_part_name}.txt'),
              'w') as fp:
        [fp.write(item + '\n') for item in real_train_item_list]
    with open(osp.join(root_path, f'{split}_{test_part_name}.txt'), 'w') as fp:
        [fp.write(item + '\n') for item in real_test_item_list]

    if val_item_list is not None:
        with open(osp.join(root_path, f'{split}_{test_part_name}.txt'),
                  'w') as fp:
            [fp.write(item + '\n') for item in real_test_item_list]


if __name__ == '__main__':
    main()
