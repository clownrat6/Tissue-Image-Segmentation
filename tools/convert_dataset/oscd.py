"""image w/o instance annotations.

image_id: image_filename
161: net (13004).jpg
293: net (6405).jpg
609: net (665).jpg
796: net (8577).jpg
1288: net (4403).jpg
1449: net (2006).jpg
1562: net (12506).jpg
1902: net (4007).jpg
2278: net (9182).jpg
3438: net (904).jpg
3476: net (4016).jpg
4264: net (2003).jpg
4695: net (11398).jpg
5116: net (12548).jpg
5208: net (3325).jpg
5566: net (3511).jpg
6290: net (11800).jpg
6436: net (9181).jpg
6587: net (2218).jpg
6698: net (3555).jpg
7102: net (4012).jpg

image has error polygon

image_id: image_filename
4858: net (2941).jpg
5356: net (1909).jpg
"""

import argparse
import json
import os
import os.path as osp
import random
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from pycocotools import _mask as mask
from skimage import morphology

cyan_color_list = [
    (0, 0, 0),
    (41, 171, 135),
    (176, 220, 213),
    (127, 200, 255),
    (0, 127, 255),
    (50, 18, 122),
]

CLASSES = {0: 'Background', 1: 'Carton'}


def polygon_to_mask(polygon, height, width, path=None):
    if len(polygon[0]) == 4:
        polygon = np.array(polygon, dtype=np.float64)
    rle_mask = mask.frPyObjects(polygon, height, width)
    ann_mask = mask.decode(rle_mask)
    # one mask contains only one instance, but one instance may be split into
    # several parts.
    ann_mask = np.max(ann_mask, axis=2)

    ann_mask_img = Image.fromarray(ann_mask.astype(np.uint8)).convert('P')

    # set palette
    color = cyan_color_list[random.randint(1, len(cyan_color_list) - 1)]
    palette = np.array([(0, 0, 0), color], dtype=np.uint8)
    ann_mask_img.putpalette(palette)

    if path is not None:
        ann_mask_img.save(path)

    return ann_mask_img


def pillow_save(image, path=None, palette=None):
    image = Image.fromarray(image)

    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)

    if path is not None:
        image.save(path)

    return image


def convert_single_image(image_id, image_dict, instance_dict,
                         image_to_instances_dict, src_image_folder,
                         image_folder, ann_folder):
    image_item = image_dict[image_id]
    if image_id in image_to_instances_dict:
        instance_ids = image_to_instances_dict[image_id]
    else:
        instance_ids = []

    image_filename = image_item['file_name']
    image_name = osp.splitext(image_filename)[0]
    height, width = image_item['height'], image_item['width']

    instances_json = {
        'imgHeight': height,
        'imgWidth': width,
        'imgName': image_name,
        'objects': []
    }

    instance_canvas = np.zeros((height, width), dtype=np.int64)
    semantic_canvas = np.zeros((height, width), dtype=np.uint8)
    for idx, instance_id in enumerate(instance_ids):
        instance_item = instance_dict[instance_id]
        # extract polygons & bounding boxes
        object_json = {}
        object_json['label'] = CLASSES[instance_item['category_id']]
        object_json['polygon'] = instance_item['segmentation']
        object_json['bbox'] = instance_item['bbox']
        # polygon may have error format.
        if len(object_json['polygon'][0]) == 4:
            continue
        instances_json['objects'].append(object_json)

        # instance mask conversion
        instance_mask = np.array(
            polygon_to_mask(instance_item['segmentation'], height, width))

        instance_canvas[instance_mask > 0] = idx + 1
        semantic_canvas[instance_mask > 0] += 1
        bound = morphology.dilation(
            instance_mask, morphology.selem.disk(1)) & (
                ~morphology.erosion(instance_mask, morphology.selem.disk(1)))
        semantic_canvas[bound > 0] = 2

    # save instance label & semantic label & polygon
    instance_ann_filename = image_name + '_instance.npy'
    semantic_ann_filename = image_name + '_semantic_with_edge.png'
    semantic_palette = np.array([(0, 0, 0), (255, 2, 255), (2, 255, 255)],
                                dtype=np.uint8)
    polygon_ann_filename = image_name + '_polygon.json'

    np.save(osp.join(ann_folder, instance_ann_filename), instance_canvas)

    pillow_save(
        semantic_canvas,
        path=osp.join(ann_folder, semantic_ann_filename),
        palette=semantic_palette)

    with open(osp.join(ann_folder, polygon_ann_filename), 'w') as fp:
        fp.write(json.dumps(instances_json, indent=4))

    # save image
    src_image_path = osp.join(src_image_folder, image_filename)
    image_path = osp.join(image_folder, image_filename)
    shutil.copy(src_image_path, image_path)

    return image_name


def parse_args():
    parser = argparse.ArgumentParser('SCD dataset conversion')
    parser.add_argument('dataset_root', help='The root path of dataset.')
    parser.add_argument(
        '-a',
        '--ann-folder',
        default=None,
        help='The annotation save folder path')
    parser.add_argument(
        '-n', '--nproc', default=1, type=int, help='The number of process.')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    split_list = ['train', 'val']

    for split in split_list:
        # image source & annotation source
        src_image_folder = osp.join(
            dataset_root,
            f'OSCD/coco_carton/oneclass_carton/images/{split}2017/')
        src_ann_json = osp.join(
            dataset_root, ('OSCD/coco_carton/oneclass_carton/annotations/'
                           f'instances_{split}2017.json'))

        image_folder = osp.join(dataset_root, 'images')
        ann_folder = args.ann_folder or osp.join(dataset_root, 'annotations')

        if not osp.exists(image_folder):
            os.makedirs(image_folder, 0o775)

        if not osp.exists(ann_folder):
            os.makedirs(ann_folder, 0o775)

        src_ann_json = json.load(open(src_ann_json, 'r'))

        images = src_ann_json['images']
        instances = src_ann_json['annotations']

        # make hash map between image and instances
        image_dict = {}
        for image in images:
            image_dict[image['id']] = image
        instance_dict = {}
        image_to_instances_dict = {}
        for instance in instances:
            instance_id = instance['id']
            related_image_id = instance['image_id']

            instance_dict[instance_id] = instance
            if related_image_id not in image_to_instances_dict:
                image_to_instances_dict[related_image_id] = [instance_id]
            else:
                image_to_instances_dict[related_image_id].append(instance_id)

        # define single loop job
        loop_job = partial(
            convert_single_image,
            image_dict=image_dict,
            instance_dict=instance_dict,
            image_to_instances_dict=image_to_instances_dict,
            src_image_folder=src_image_folder,
            image_folder=image_folder,
            ann_folder=ann_folder,
        )

        image_ids = image_dict.keys()
        miss_ids = []
        image_names = []
        # check existed images
        for image_id in image_ids:
            image_item = image_dict[image_id]
            image_filename = image_item['file_name']
            image_name = osp.splitext(image_filename)[0]
            image_path = osp.join(image_folder, image_filename)
            if not osp.exists(image_path):
                miss_ids.append(image_id)
            else:
                image_names.append(image_name)

        # only build miss images & labels
        if args.nproc > 1:
            records = mmcv.track_parallel_progress(loop_job, miss_ids,
                                                   args.nproc)
        else:
            records = mmcv.track_progress(loop_job, miss_ids)

        image_names.extend(records)

        with open(f'{dataset_root}/{split}.txt', 'w') as fp:
            for image_name in image_names:
                fp.write(image_name + '\n')


if __name__ == '__main__':
    main()
