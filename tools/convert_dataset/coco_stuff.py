import argparse
import json
import os
import os.path as osp
import shutil
from functools import partial

import mmcv
import numpy as np
from PIL import Image
from pycocotools import _mask as mask
from pycocotools import coco
from skimage import morphology

# coco official id & label name
CLASSES = {
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
    92: 'banner',
    93: 'blanket',
    94: 'branch',
    95: 'bridge',
    96: 'building-other',
    97: 'bush',
    98: 'cabinet',
    99: 'cage',
    100: 'cardboard',
    101: 'carpet',
    102: 'ceiling-other',
    103: 'ceiling-tile',
    104: 'cloth',
    105: 'clothes',
    106: 'clouds',
    107: 'counter',
    108: 'cupboard',
    109: 'curtain',
    110: 'desk-stuff',
    111: 'dirt',
    112: 'door-stuff',
    113: 'fence',
    114: 'floor-marble',
    115: 'floor-other',
    116: 'floor-stone',
    117: 'floor-tile',
    118: 'floor-wood',
    119: 'flower',
    120: 'fog',
    121: 'food-other',
    122: 'fruit',
    123: 'furniture-other',
    124: 'grass',
    125: 'gravel',
    126: 'ground-other',
    127: 'hill',
    128: 'house',
    129: 'leaves',
    130: 'light',
    131: 'mat',
    132: 'metal',
    133: 'mirror-stuff',
    134: 'moss',
    135: 'mountain',
    136: 'mud',
    137: 'napkin',
    138: 'net',
    139: 'paper',
    140: 'pavement',
    141: 'pillow',
    142: 'plant-other',
    143: 'plastic',
    144: 'platform',
    145: 'playingfield',
    146: 'railing',
    147: 'railroad',
    148: 'river',
    149: 'road',
    150: 'rock',
    151: 'roof',
    152: 'rug',
    153: 'salad',
    154: 'sand',
    155: 'sea',
    156: 'shelf',
    157: 'sky-other',
    158: 'skyscraper',
    159: 'snow',
    160: 'solid-other',
    161: 'stairs',
    162: 'stone',
    163: 'straw',
    164: 'structural-other',
    165: 'table',
    166: 'tent',
    167: 'textile-other',
    168: 'towel',
    169: 'tree',
    170: 'vegetable',
    171: 'wall-brick',
    172: 'wall-concrete',
    173: 'wall-other',
    174: 'wall-panel',
    175: 'wall-stone',
    176: 'wall-tile',
    177: 'wall-wood',
    178: 'water-other',
    179: 'waterdrops',
    180: 'window-blind',
    181: 'window-other',
    182: 'wood',
    255: 'ignore',
    256: 'edge'
}

# convert coco official cat id to new id
LABEL_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    92: 80,
    93: 81,
    94: 82,
    95: 83,
    96: 84,
    97: 85,
    98: 86,
    99: 87,
    100: 88,
    101: 89,
    102: 90,
    103: 91,
    104: 92,
    105: 93,
    106: 94,
    107: 95,
    108: 96,
    109: 97,
    110: 98,
    111: 99,
    112: 100,
    113: 101,
    114: 102,
    115: 103,
    116: 104,
    117: 105,
    118: 106,
    119: 107,
    120: 108,
    121: 109,
    122: 110,
    123: 111,
    124: 112,
    125: 113,
    126: 114,
    127: 115,
    128: 116,
    129: 117,
    130: 118,
    131: 119,
    132: 120,
    133: 121,
    134: 122,
    135: 123,
    136: 124,
    137: 125,
    138: 126,
    139: 127,
    140: 128,
    141: 129,
    142: 130,
    143: 131,
    144: 132,
    145: 133,
    146: 134,
    147: 135,
    148: 136,
    149: 137,
    150: 138,
    151: 139,
    152: 140,
    153: 141,
    154: 142,
    155: 143,
    156: 144,
    157: 145,
    158: 146,
    159: 147,
    160: 148,
    161: 149,
    162: 150,
    163: 151,
    164: 152,
    165: 153,
    166: 154,
    167: 155,
    168: 156,
    169: 157,
    170: 158,
    171: 159,
    172: 160,
    173: 161,
    174: 162,
    175: 163,
    176: 164,
    177: 165,
    178: 166,
    179: 167,
    180: 168,
    181: 169,
    182: 170,
    255: 255,
    256: 171
}

PALETTE = []

EDGE_ID = 256


def polygon_to_mask(polygon, height, width, path=None, if_rle=False):
    # XXX: When len(polygon[0]) == 4, the polygon annotation is recognized as
    # a bounding box rather than a polygon.
    # if len(polygon[0]) == 4:
    #     polygon = np.array(polygon, dtype=np.float64)

    if not if_rle:
        rle_mask = mask.frPyObjects(polygon, height, width)
    else:
        rle_mask = polygon
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
        if not isinstance(palette, np.ndarray):
            palette = np.array(palette, dtype=np.uint8)
        image.putpalette(palette)

    if path is not None:
        image.save(path)

    return image


cat_count = {}


def convert_single_image(task, ann_folder, put_palette=False):
    img_item, anns = task
    things_anns, stuff_anns = anns

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
    semantic_canvas.fill(255)
    semantic_edge_canvas = np.zeros((height, width), dtype=np.uint8)
    semantic_edge_canvas.fill(255)

    # stuff classes drawing
    for idx, ann in enumerate(stuff_anns):
        stuff_item = ann
        # iscrowd is not considered in coco stuff segmentation dataset.
        if stuff_item['iscrowd'] == 1:
            continue
        cat_id = stuff_item['category_id']
        # ignore other classes when converting stuff mask.
        if cat_id == 183:
            continue
        # the raw category ids aren't continuous, so we use label_map to let
        # them continous.
        new_id = LABEL_MAP[cat_id]
        # extract polygons & bounding boxes
        object_json = {}
        object_json['label'] = CLASSES[cat_id]
        object_json['polygon'] = stuff_item['segmentation']
        object_json['bbox'] = stuff_item['bbox']
        # polygon may have error format.
        if isinstance(object_json['polygon'], list) and len(
                object_json['polygon'][0]) == 4:
            continue
        instances_json['objects'].append(object_json)

        # instance mask conversion
        stuff_mask = polygon_to_mask(
            stuff_item['segmentation'], height, width, if_rle=True)
        semantic_canvas[stuff_mask > 0] = new_id
        semantic_edge_canvas[stuff_mask > 0] = new_id

    # things classes drawing
    for idx, ann in enumerate(things_anns):
        instance_item = ann
        # iscrowd is not considered in coco stuff segmentation dataset.
        if instance_item['iscrowd'] == 1:
            continue
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
        semantic_edge_canvas[instance_mask > 0] = new_id
        bound = morphology.dilation(
            instance_mask, morphology.selem.disk(1)) & (
                ~morphology.erosion(instance_mask, morphology.selem.disk(1)))
        semantic_edge_canvas[bound > 0] = LABEL_MAP[EDGE_ID]

    # save instance label & semantic label & polygon
    instance_ann_filename = img_name + '_instance.npy'
    semantic_ann_filename = img_name + '_semantic.png'
    semantic_edge_ann_filename = img_name + '_semantic_with_edge.png'
    polygon_ann_filename = img_name + '_polygon.json'

    np.save(osp.join(ann_folder, instance_ann_filename), instance_canvas)

    if put_palette:
        palette = PALETTE
    else:
        palette = None

    pillow_save(
        semantic_canvas,
        path=osp.join(ann_folder, semantic_ann_filename),
        palette=palette)

    pillow_save(
        semantic_edge_canvas,
        path=osp.join(ann_folder, semantic_edge_ann_filename),
        palette=palette)

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
        '-p', '--put-palette', help='Whether to put palette for sementic map.')
    parser.add_argument(
        '-n', '--nproc', default=1, type=int, help='The number of process.')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    split_list = ['train', 'val']

    for split in split_list:
        # image source & annotation source
        things_json = osp.join(dataset_root,
                               f'coco/annotations/instances_{split}2017.json')
        stuff_json = osp.join(dataset_root,
                              f'coco/annotations/stuff_{split}2017.json')

        ann_folder = args.ann_folder or osp.join(dataset_root, 'annotations')

        things_json = coco.COCO(things_json)
        stuff_json = coco.COCO(stuff_json)

        if args.re_generate:
            shutil.rmtree(ann_folder)

        if not osp.exists(ann_folder):
            os.makedirs(ann_folder, 0o775)

        img_ids = things_json.getImgIds()

        # Whether re-generate whole dataset or not.
        if args.re_generate:
            miss_ids = img_ids
            img_names = []
        else:
            miss_ids = []
            img_names = []
            img_items = things_json.loadImgs(ids=img_ids)
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
        imgs = things_json.loadImgs(ids=miss_ids)
        img_related_anns = []
        for img_id in miss_ids:
            thing_anns = [
                things_json.loadAnns(ids=[ann_id])[0]
                for ann_id in things_json.getAnnIds(imgIds=[img_id])
            ]
            stuff_anns = [
                stuff_json.loadAnns(ids=[ann_id])[0]
                for ann_id in stuff_json.getAnnIds(imgIds=[img_id])
            ]
            img_related_anns.append((thing_anns, stuff_anns))

        tasks = list(zip(imgs, img_related_anns))

        # define single loop job
        loop_job = partial(
            convert_single_image,
            ann_folder=ann_folder,
            put_palette=args.put_palette,
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
