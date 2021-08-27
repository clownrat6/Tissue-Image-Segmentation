import argparse
import math
import os
import os.path as osp
import shutil

import cv2
import numpy as np
from lxml import etree
from PIL import Image
from rich.progress import track
from skimage import morphology


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


def convert_train_cohort(raw_path, new_path, with_edge=True):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    raw_image_folder = osp.join(raw_path, 'Tissue Images')
    raw_label_folder = osp.join(raw_path, 'Annotations')

    item_list = [x.rstrip('.tif') for x in os.listdir(raw_image_folder)]

    for item in track(item_list):
        image_path = osp.join(raw_image_folder, item + '.tif')
        label_path = osp.join(raw_label_folder, item + '.xml')

        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        contours = extract_contours(label_path)
        instance_label = convert_contour_to_instance(contours, H, W)
        semantic_label = convert_instance_to_semantic(instance_label, H, W,
                                                      len(contours), with_edge)

        # Save image
        dst_image_path = osp.join(new_path, item + '.tif')
        shutil.copy(image_path, dst_image_path)

        # Save instance level label
        dst_instance_label_path = osp.join(new_path, item + '_instance.npy')
        np.save(dst_instance_label_path, instance_label)

        # Save semantic level label
        dst_semantic_label_path = osp.join(new_path, item + '_semantic.png')
        semantic_label_png = Image.fromarray(semantic_label)
        semantic_label_png.save(dst_semantic_label_path)


def convert_test_cohort(raw_path, new_path, with_edge=True):
    if not osp.exists(new_path):
        os.makedirs(new_path, 0o775)

    item_list = [x.rstrip('.tif') for x in os.listdir(raw_path) if '.tif' in x]

    for item in track(item_list):
        image_path = osp.join(raw_path, item + '.tif')
        label_path = osp.join(raw_path, item + '.xml')

        image = cv2.imread(image_path)
        H, W = image.shape[:2]
        contours = extract_contours(label_path)
        instance_label = convert_contour_to_instance(contours, H, W)
        semantic_label = convert_instance_to_semantic(instance_label, H, W,
                                                      len(contours), with_edge)

        # Save image
        dst_image_path = osp.join(new_path, item + '.tif')
        shutil.copy(image_path, dst_image_path)

        # Save instance level label
        dst_instance_label_path = osp.join(new_path, item + '_instance.npy')
        np.save(dst_instance_label_path, instance_label)

        # Save semantic level label
        dst_semantic_label_path = osp.join(new_path, item + '_semantic.png')
        semantic_label_png = Image.fromarray(semantic_label)
        semantic_label_png.save(dst_semantic_label_path)


def parse_args():
    parser = argparse.ArgumentParser('Convert monuseg dataset.')
    parser.add_argument('root_path', help='dataset root path.')
    parser.add_argument(
        '-e',
        '--with_edge',
        action='store_true',
        help='whether to make semantic label with edge class.')

    return parser.parse_args()


def main():
    args = parse_args()
    root_path = args.root_path
    with_edge = args.with_edge

    train_raw_path = osp.join(root_path, 'MoNuSeg 2018 Training Data',
                              'MoNuSeg 2018 Training Data')
    test_raw_path = osp.join(root_path, 'MoNuSegTestData')

    train_new_path = osp.join(root_path, 'train')
    test_new_path = osp.join(root_path, 'test')

    # make train cohort dataset
    convert_train_cohort(train_raw_path, train_new_path, with_edge)

    # make test cohort dataset
    convert_test_cohort(test_raw_path, test_new_path, with_edge)


if __name__ == '__main__':
    main()
