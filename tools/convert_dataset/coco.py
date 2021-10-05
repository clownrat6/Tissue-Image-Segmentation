import argparse
import json
import os
import os.path as osp
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from pycocotools import _mask as mask
from pycocotools import coco
from skimage import morphology

# coco official id & label name
CLASSES = {
    0: 'Background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    255: 'edge'
}

# convert coco official id to new id
LABEL_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    27: 25,
    28: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    65: 60,
    67: 61,
    70: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    82: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    90: 80,
    255: 81,
}


def polygon_to_mask(polygon, height, width, path=None):
    # XXX: When len(polygon[0]) == 4, the polygon annotation is recognized as
    # a bounding box rather than a polygon.
    # if len(polygon[0]) == 4:
    #     polygon = np.array(polygon, dtype=np.float64)

    rle_mask = mask.frPyObjects(polygon, height, width)
    if not isinstance(rle_mask, list):
        rle_mask = [rle_mask]
    ann_mask = mask.decode(rle_mask)

    # one mask contains only one instance, but one instance may be split into
    # several parts.
    ann_mask = np.max(ann_mask, axis=2)

    if path is not None:
        ann_mask_img = Image.fromarray(ann_mask.astype(np.uint8))
        ann_mask_img.save(path)

    return ann_mask


def pillow_save(image, path=None, palette=None):
    image = Image.fromarray(image)

    if palette is not None:
        image = image.convert('P')
        image.putpalette(palette)

    if path is not None:
        image.save(path)

    return image


def convert_single_image(task, ann_folder):
    img_item, anns = task

    img_name = osp.splitext(img_item['file_name'])[0]
    height, width = img_item['height'], img_item['width']

    instances_json = {
        'imgHeight': height,
        'imgWidth': width,
        'imgName': img_name,
        'objects': []
    }

    instance_canvas = np.zeros((height, width), dtype=np.int64)
    semantic_canvas = np.zeros((height, width), dtype=np.uint8)
    semantic_edge_canvas = np.zeros((height, width), dtype=np.uint8)
    for idx, ann in enumerate(anns):
        instance_item = ann
        cat_id = instance_item['category_id']
        # the raw category ids aren't continuous, so we use label_map to let
        # them continous.
        new_id = LABEL_MAP[cat_id]
        # extract polygons & bounding boxes
        object_json = {}
        object_json['label'] = CLASSES[cat_id]
        object_json['polygon'] = instance_item['segmentation']
        object_json['bbox'] = instance_item['bbox']

        # polygon may have error format.
        if isinstance(object_json['polygon'], list) and len(
                object_json['polygon'][0]) == 4:
            continue
        instances_json['objects'].append(object_json)

        # instance mask conversion
        instance_mask = polygon_to_mask(instance_item['segmentation'], height,
                                        width)

        instance_canvas[instance_mask > 0] = new_id * 1000 + idx + 1
        semantic_canvas[instance_mask > 0] = new_id
        semantic_edge_canvas = semantic_canvas.copy()
        bound = morphology.dilation(
            instance_mask, morphology.selem.disk(1)) & (
                ~morphology.erosion(instance_mask, morphology.selem.disk(1)))
        semantic_edge_canvas[bound > 0] = LABEL_MAP[255]

    # save instance label & semantic label & polygon
    instance_ann_filename = img_name + '_instance.npy'
    semantic_ann_filename = img_name + '_semantic.png'
    semantic_edge_ann_filename = img_name + '_semantic_with_edge.png'
    polygon_ann_filename = img_name + '_polygon.json'

    np.save(osp.join(ann_folder, instance_ann_filename), instance_canvas)

    pillow_save(
        semantic_canvas, path=osp.join(ann_folder, semantic_ann_filename))

    pillow_save(
        semantic_edge_canvas,
        path=osp.join(ann_folder, semantic_edge_ann_filename))

    with open(osp.join(ann_folder, polygon_ann_filename), 'w') as fp:
        fp.write(json.dumps(instances_json, indent=4))

    return img_name


def parse_args():
    parser = argparse.ArgumentParser('coco dataset conversion')
    parser.add_argument('dataset_root', help='The root path of dataset.')
    parser.add_argument(
        '-r',
        '--re-generate',
        action='store_true',
        help='restart a dataset conversion.')
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

    split_list = ['val']

    for split in split_list:
        # image source & annotation source
        src_ann_json = osp.join(
            dataset_root, f'coco/annotations/instances_{split}2017.json')
        ann_folder = args.ann_folder or osp.join(dataset_root, 'annotations')

        src_ann_json = coco.COCO(src_ann_json)

        if not osp.exists(ann_folder):
            os.makedirs(ann_folder, 0o775)

        img_ids = src_ann_json.getImgIds()

        # Whether re-generate whole dataset or not.
        if args.re_generate:
            miss_ids = img_ids
            img_names = []
        else:
            miss_ids = []
            img_names = []
            img_items = src_ann_json.loadImgs(ids=img_ids)
            # check existed images
            for img_id, img_item in zip(img_ids, img_items):
                img_filename = img_item['file_name']
                img_name = osp.splitext(img_filename)[0]
                ann_path = osp.join(ann_folder, img_name + '_instance.npy')
                if not osp.exists(ann_path):
                    miss_ids.append(img_id)
                else:
                    img_names.append(img_name)

        # make multi-threads loop tasks
        imgs = src_ann_json.loadImgs(ids=miss_ids)
        img_related_anns = [[
            src_ann_json.loadAnns(ids=[ann_id])[0]
            for ann_id in src_ann_json.getAnnIds(imgIds=[img_id])
        ] for img_id in miss_ids]

        tasks = list(zip(imgs, img_related_anns))

        # define single loop job
        loop_job = partial(
            convert_single_image,
            ann_folder=ann_folder,
        )

        if args.nproc > 1:
            records = mmcv.track_parallel_progress(
                loop_job, tasks, nproc=args.nproc)
        else:
            records = mmcv.track_progress(loop_job, tasks)

        img_names.extend(records)

        with open(f'{dataset_root}/{split}.txt', 'w') as fp:
            for img_name in img_names:
                fp.write(img_name + '\n')


if __name__ == '__main__':
    main()
