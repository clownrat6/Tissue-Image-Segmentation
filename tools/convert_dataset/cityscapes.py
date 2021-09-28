import argparse
import os.path as osp
import re

import mmcv
import numpy as np
from cityscapesscripts.preparation.json2instanceImg import json2instanceImg
from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image
from skimage import morphology

label_id_map = {
    'person': {
        'official': 11,
        'local': 1
    },
    'rider': {
        'official': 12,
        'local': 2
    },
    'car': {
        'official': 13,
        'local': 3
    },
    'truck': {
        'official': 14,
        'local': 4
    },
    'bus': {
        'official': 15,
        'local': 5
    },
    'train': {
        'official': 16,
        'local': 6
    },
    'motocycle': {
        'official': 17,
        'local': 7
    },
    'bicycle': {
        'official': 18,
        'local': 8
    },
    'edge': {
        'official': None,
        'local': 9
    }
}


def pillow_load(path):
    return np.array(Image.open(path))


def pillow_save(array, path):
    Image.fromarray(array).save(path)
    return path


def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')
    return label_file


def convert_json_to_instance(json_file):
    instance_file = json_file.replace('_polygons.json',
                                      '_instanceTrainIds.png')
    json2instanceImg(json_file, instance_file, 'trainIds')
    return instance_file


def convert_official_to_local(official_id):
    for id_map in label_id_map.values():
        if id_map['official'] == official_id:
            return id_map['local']
    return 0


def convert_instance_to_semantic(instances, with_edge=True):
    """Convert instance mask to semantic mask.

    Args:
        instances (numpy.ndarray): The mask contains each instances with
            different label value.
        with_edge (bool): Convertion with edge class label. Default: True.

    Returns:
        mask (numpy.ndarray): instance related semantic map with edge or not.
            (background, car, person, ..., edge)
    """
    save_path = None
    if isinstance(instances, str):
        save_path = re.compile('.*gtFine').findall(
            instances)[0] + '_semantic_with_edge.png'
        instances = pillow_load(instances)
    height, width = instances.shape[:2]
    mask = np.zeros([height, width], dtype=np.uint8)
    instance_id_list = list(np.unique(instances))
    for instance_id in instance_id_list:
        single_instance_map = (instances == instance_id).astype(np.uint8)
        if int(instance_id / 1000) == 255:
            continue
        local_id = convert_official_to_local(int(instance_id / 1000))
        mask[single_instance_map > 0] = local_id
        if with_edge:
            boundary = morphology.dilation(
                single_instance_map,
                morphology.selem.disk(1)) & (~morphology.erosion(
                    single_instance_map, morphology.selem.disk(1)))
            # the official id of edge is None.
            # official dataset doesn't have edge class.
            mask[boundary > 0] = convert_official_to_local(None)
    if save_path is not None:
        return pillow_save(mask, save_path)
    else:
        return mask


def semantic_label_conversion(gt_dir, out_dir, nproc=1):
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    if nproc > 1:
        mmcv.track_parallel_progress(convert_json_to_label, poly_files, nproc)
    else:
        mmcv.track_progress(convert_json_to_label, poly_files)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


def instance_label_conversion(gt_dir, out_dir, nproc=1):
    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    if nproc > 1:
        instance_files = mmcv.track_parallel_progress(convert_json_to_instance,
                                                      poly_files, nproc)
        mmcv.track_parallel_progress(convert_instance_to_semantic,
                                     instance_files, nproc)
    else:
        instance_files = mmcv.track_progress(convert_json_to_instance,
                                             poly_files)
        mmcv.track_progress(convert_instance_to_semantic, instance_files,
                            nproc)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


def semantic_edge_label_conversion(gt_dir, out_dir, nproc=1):
    instance_files = []
    for instance in mmcv.scandir(
            gt_dir, '_instanceTrainIds.png', recursive=True):
        instance_file = osp.join(gt_dir, instance)
        instance_files.append(instance_file)
    if nproc > 1:
        mmcv.track_parallel_progress(convert_instance_to_semantic,
                                     instance_files, nproc)
    else:
        mmcv.track_progress(convert_instance_to_semantic, instance_files)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument(
        'task_level',
        help='perform semantic segmentation or instance segmentation.')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    task_level = args.task_level
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    assert task_level in ['semantic', 'instance', 'semantic_edge']

    gt_dir = osp.join(cityscapes_path, args.gt_dir)
    if task_level == 'semantic':
        semantic_label_conversion(gt_dir, out_dir, args.nproc)
    elif task_level == 'instance':
        instance_label_conversion(gt_dir, out_dir, args.nproc)
    elif task_level == 'semantic_edge':
        semantic_edge_label_conversion(gt_dir, out_dir, args.nproc)


if __name__ == '__main__':
    main()
