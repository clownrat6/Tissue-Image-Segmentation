import os
import os.path as osp
import warnings
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology
from skimage.morphology import remove_small_objects
from torch.utils.data import Dataset

from tiseg.utils.evaluation.metrics import (aggregated_jaccard_index, dice_similarity_coefficient, precision_recall)
from .builder import DATASETS
from .nuclei_dataset_mapper import NucleiDatasetMapper
from .utils import colorize_seg_map, re_instance, mudslide_watershed, align_foreground


def draw_semantic(save_folder, data_id, image, pred, label, edge_id=2):
    """draw semantic level picture with FP & FN."""

    plt.figure(figsize=(5 * 2, 5 * 2 + 3))

    # prediction drawing
    plt.subplot(221)
    plt.imshow(pred)
    plt.axis('off')
    plt.title('Prediction', fontsize=15, color='black')

    # ground truth drawing
    plt.subplot(222)
    plt.imshow(label)
    plt.axis('off')
    plt.title('Ground Truth', fontsize=15, color='black')

    # image drawing
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(223)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image', fontsize=15, color='black')

    canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
    canvas[label > 0, :] = (255, 255, 2)
    canvas[canvas == edge_id] = 0
    canvas[(pred == 0) * (label > 0), :] = (2, 255, 255)
    canvas[(pred > 0) * (label == 0), :] = (255, 2, 255)
    plt.subplot(224)
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('FN-FP-Ground Truth', fontsize=15, color='black')

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [(255, 255, 2), (2, 255, 255), (255, 2, 255)]
    label_list = [
        'Ground Truth',
        'FN',
        'FP',
    ]
    for color, label in zip(colors, label_list):
        color = list(color)
        color = [x / 255 for x in color]
        plt.plot(0, 0, '-', color=tuple(color), label=label)
    plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    # results visulization
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{data_id}_semantic_compare.png', dpi=300)


def draw_instance(save_folder, data_id, pred_instance, label_instance):
    """draw instance level picture."""

    plt.figure(figsize=(5 * 2, 5))

    plt.subplot(121)
    plt.imshow(colorize_seg_map(pred_instance))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(colorize_seg_map(label_instance))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_folder}/{data_id}_instance_compare.png', dpi=300)


@DATASETS.register_module()
class NucleiCustomDataset(Dataset):
    """Nuclei Custom Foundation Segmentation Dataset.

    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, Nuclei1, Nuclei2, ...).
    The basic settings only supports two-class nuclei segmentation task.

    related suffix:
        "_semantic_with_edge.png": three class semantic map with edge.
        "_semantic.png": raw semantic map (two class semantic map without
            boundary).
        "_instance.npy": instance level map.
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self,
                 process_cfg,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 img_suffix='.tif',
                 sem_suffix='_semantic.png',
                 inst_suffix='_instance.npy',
                 test_mode=False,
                 split=None):

        self.mapper = NucleiDatasetMapper(test_mode, process_cfg=process_cfg)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.data_root = data_root

        self.img_suffix = img_suffix
        self.sem_suffix = sem_suffix
        self.inst_suffix = inst_suffix

        self.test_mode = test_mode
        self.split = split

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.data_infos = self.load_annotations(self.img_dir, self.ann_dir, self.img_suffix, self.sem_suffix,
                                                self.inst_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def __getitem__(self, index):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        data_info = self.data_infos[index]
        return self.mapper(data_info)

    def load_annotations(self, img_dir, ann_dir, img_suffix, sem_suffix, inst_suffix, split=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory.
            ann_dir (str): Path to annotation directory.
            img_suffix (str): Suffix of images.
            ann_suffix (str): Suffix of segmentation maps.
            split (str | None): Split txt file. If split is specified, only
                file with suffix in the splits will be loaded.

        Returns:
            list[dict]: All data info of dataset, data info contains image,
                segmentation map.
        """
        data_infos = []
        if split is not None:
            with open(split, 'r') as fp:
                for line in fp.readlines():
                    img_id = line.strip()
                    img_name = img_id + img_suffix
                    sem_name = img_id + sem_suffix
                    inst_name = img_id + inst_suffix
                    img_file_name = osp.join(img_dir, img_name)
                    sem_file_name = osp.join(ann_dir, sem_name)
                    inst_file_name = osp.join(ann_dir, inst_name)
                    data_info = dict(
                        file_name=img_file_name, sem_file_name=sem_file_name, inst_file_name=inst_file_name)
                    data_infos.append(data_info)
        else:
            for img_name in mmcv.scandir(img_dir, img_suffix, recursive=True):
                sem_name = img_name.replace(img_suffix, sem_suffix)
                inst_name = img_name.replace(img_suffix, inst_suffix)
                img_file_name = osp.join(img_dir, img_name)
                sem_file_name = osp.join(ann_dir, sem_name)
                inst_file_name = osp.join(ann_dir, inst_name)
                data_info = dict(file_name=img_file_name, sem_file_name=sem_file_name, inst_file_name=inst_file_name)
                data_infos.append(data_info)

        return data_infos

    def pre_eval(self, preds, indices, show_semantic=False, show_instance=False, show_folder=None):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            show_semantic (bool): Illustrate semantic level prediction &
                ground truth. Default: False
            show_instance (bool): Illustrate instance level prediction &
                ground truth. Default: False
            show_folder (str | None, optional): The folder path of
                illustration. Default: None

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        if show_folder is None and (show_semantic or show_instance):
            warnings.warn('show_semantic or show_instance is set to True, but the '
                          'show_folder is None. We will use default show_folder: '
                          '.nuclei_show')
            show_folder = '.nuclei_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_seg = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_seg = np.load(inst_file_name)
            inst_seg = re_instance(inst_seg)

            data_id = osp.basename(self.data_infos[index]['sem_file_name']).replace(self.sem_suffix, '')

            # metric calculation post process codes:
            # extract inside
            sem_pred = pred['sem_pred']
            fore_pred = (sem_pred > 0).astype(np.uint8)
            sem_pred_in = (sem_pred == 1).astype(np.uint8)

            if 'dir_pred' in pred:
                dir_pred = pred['dir_pred']

                # model-agnostic post process operations
                sem_pred, inst_pred, fore_pred = self.model_agnostic_postprocess_w_dir(sem_pred_in, fore_pred, dir_pred)
            else:
                sem_pred, inst_pred = self.model_agnostic_postprocess(sem_pred_in)

            # TODO: (Important issue about post process)
            # This may be the dice metric calculation trick (Need be
            # considering carefully)
            # convert instance map (after postprocess) to semantic level
            sem_pred = (inst_pred > 0).astype(np.uint8)

            # semantic metric calculation (remove background class)
            # [1] will remove background class.
            precision_metric, recall_metric = precision_recall(sem_pred, sem_seg, 2)
            precision_metric = precision_metric[1]
            recall_metric = recall_metric[1]
            dice_metric = dice_similarity_coefficient(sem_pred, sem_seg, 2)[1]

            # instance metric calculation
            aji_metric = aggregated_jaccard_index(inst_pred, inst_seg, is_semantic=False)

            single_loop_results = dict(
                name=data_id,
                Aji=aji_metric,
                Dice=dice_metric,
                Recall=recall_metric,
                Precision=precision_metric,
            )
            pre_eval_results.append(single_loop_results)

            # illustrating semantic level results
            if show_semantic:
                data_info = self.data_infos[index]
                draw_semantic(show_folder, data_id, data_info['file_name'], sem_pred, sem_seg, single_loop_results)

            # illustrating instance level results
            if show_instance:
                draw_instance(show_folder, data_id, inst_pred, inst_seg)

        return pre_eval_results

    def model_agnostic_postprocess(self, sem_pred_in):
        """model free post-process for both instance-level & semantic-level."""
        # fill instance holes
        sem_pred = binary_fill_holes(sem_pred_in)
        # remove small instance
        sem_pred = remove_small_objects(sem_pred, 20)
        sem_pred = sem_pred.astype(np.uint8)

        # instance process & dilation
        inst_pred = measure.label(sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))

        return sem_pred, inst_pred

    def model_agnostic_postprocess_w_dir(self, sem_pred_in, fore_pred, dir_pred):
        """model free post-process for both instance-level & semantic-level."""
        raw_fore_pred = fore_pred
        # fill instance holes
        sem_pred = binary_fill_holes(sem_pred_in)
        # remove small instance
        sem_pred = remove_small_objects(sem_pred, 20)
        sem_pred = sem_pred.astype(np.uint8)

        # instance process & dilation
        fore_pred = measure.label(sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        fore_pred = morphology.dilation(fore_pred, selem=morphology.selem.disk(2))

        raw_fore_pred = binary_fill_holes(raw_fore_pred)  # 孔洞填充 hhl20200414
        raw_fore_pred = remove_small_objects(raw_fore_pred, 20)  # remove small object

        sem_pred, bound = mudslide_watershed(sem_pred, dir_pred, raw_fore_pred)

        sem_pred = remove_small_objects(sem_pred, 20)
        inst_pred = measure.label(sem_pred, connectivity=1)
        inst_pred = align_foreground(inst_pred, fore_pred, 20)

        return sem_pred, inst_pred, fore_pred

    def evaluate(self, results, metric='all', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'Aji',
                'Dice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        if 'all' in metric:
            metric = ['IoU', 'Dice', 'Precision', 'Recall']
        allowed_metrics = ['IoU', 'Dice', 'Precision', 'Recall']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        ret_metrics = {}
        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # calculate average metric
        assert 'name' in ret_metrics
        name_list = ret_metrics.pop('name')
        name_list.append('Average')
        for key in ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            # average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
            if key in metric:
                average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
            elif key == 'Aji':
                average_value = sum(ret_metrics[key]) / len(ret_metrics[key])

            ret_metrics[key].append(average_value)
            ret_metrics[key] = np.array(ret_metrics[key])

        # for logger
        ret_metrics_items = OrderedDict(
            {ret_metric: np.round(ret_metric_value * 100, 2)
             for ret_metric, ret_metric_value in ret_metrics.items()})
        ret_metrics_items.update({'name': name_list})
        ret_metrics_items.move_to_end('name', last=False)
        items_table_data = PrettyTable()
        for key, val in ret_metrics_items.items():
            items_table_data.add_column(key, val)

        print_log('Per samples:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        eval_results = {}
        # average results
        if 'Aji' in ret_metrics:
            eval_results['Aji'] = ret_metrics['Aji'][-1]
        if 'Dice' in ret_metrics:
            eval_results['Dice'] = ret_metrics['Dice'][-1]
        if 'Recall' in ret_metrics:
            eval_results['Recall'] = ret_metrics['Recall'][-1]
        if 'Precision' in ret_metrics:
            eval_results['Precision'] = ret_metrics['Precision'][-1]

        ret_metrics_items.pop('name', None)
        for key, value in ret_metrics_items.items():
            eval_results.update({key + '.' + str(name): f'{value[idx]:.3f}' for idx, name in enumerate(name_list)})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
