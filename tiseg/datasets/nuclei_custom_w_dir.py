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

from tiseg.utils import (dice_similarity_coefficient, precision_recall, pre_eval_all_semantic_metric,
                         pre_eval_to_sem_metrics, pre_eval_bin_aji, pre_eval_aji, pre_eval_bin_pq, pre_eval_pq,
                         pre_eval_to_bin_aji, pre_eval_to_aji, pre_eval_to_bin_pq, pre_eval_to_pq,
                         pre_eval_to_sample_pq, pre_eval_to_imw_pq, pre_eval_to_imw_aji, binary_inst_dice)

from .builder import DATASETS
from .nuclei_dataset_mapper import NucleiDatasetMapper
from .utils import colorize_seg_map, re_instance, mudslide_watershed, align_foreground, assign_sem_class_to_insts, get_tc_from_inst


def draw_all(save_folder,
             img_name,
             img_file_name,
             sem_pred,
             sem_gt,
             inst_pred,
             inst_gt,
             tc_sem_pred,
             tc_sem_gt,
             edge_id=2,
             sem_palette=None):

    plt.figure(figsize=(5 * 4, 5 * 2 + 3))

    # image drawing
    img = cv2.imread(img_file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(241)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Image', fontsize=15, color='black')

    canvas = np.zeros((*sem_pred.shape, 3), dtype=np.uint8)
    canvas[(sem_pred > 0) * (sem_gt > 0), :] = (0, 0, 255)
    canvas[canvas == edge_id] = 0
    canvas[(sem_pred == 0) * (sem_gt > 0), :] = (0, 255, 0)
    canvas[(sem_pred > 0) * (sem_gt == 0), :] = (255, 0, 0)
    plt.subplot(242)
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('Error Analysis: FN-FP-TP', fontsize=15, color='black')

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    label_list = [
        'TP',
        'FN',
        'FP',
    ]
    for color, label in zip(colors, label_list):
        color = list(color)
        color = [x / 255 for x in color]
        plt.plot(0, 0, '-', color=tuple(color), label=label)
    plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    plt.subplot(243)
    plt.imshow(colorize_seg_map(inst_pred))
    plt.axis('off')
    plt.title('Instance Level Prediction')

    plt.subplot(244)
    plt.imshow(colorize_seg_map(inst_gt))
    plt.axis('off')
    plt.title('Instance Level Ground Truth')

    plt.subplot(245)
    plt.imshow(colorize_seg_map(sem_pred, sem_palette))
    plt.axis('off')
    plt.title('Semantic Level Prediction')

    plt.subplot(246)
    plt.imshow(colorize_seg_map(sem_gt, sem_palette))
    plt.axis('off')
    plt.title('Semantic Level Ground Truth')

    tc_palette = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]

    plt.subplot(247)
    plt.imshow(colorize_seg_map(tc_sem_pred, tc_palette))
    plt.axis('off')
    plt.title('Three-class Semantic Level Prediction')

    plt.subplot(248)
    plt.imshow(colorize_seg_map(tc_sem_gt, tc_palette))
    plt.axis('off')
    plt.title('Three-class Semantic Level Ground Truth')

    # results visulization
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{img_name}_compare.png', dpi=300)


@DATASETS.register_module()
class NucleiCustomDatasetWithDirection(Dataset):
    """Nuclei Custom Foundation Segmentation Dataset.
    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, Nuclei1, Nuclei2, ...).
    The basic settings only supports two-class nuclei segmentation task.
    related suffix:
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

    def pre_eval(self, preds, indices, show=False, show_folder=None):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            show (bool): Illustrate semantic level & instance level prediction &
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

        if show_folder is None and show:
            warnings.warn('show_semantic or show_instance is set to True, but the '
                          'show_folder is None. We will use default show_folder: '
                          '.nuclei_show')
            show_folder = '.nuclei_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            # img related infos
            img_file_name = self.data_infos[index]['file_name']
            img_name = osp.splitext(osp.basename(img_file_name))[0]
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_seg = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_seg = np.load(inst_file_name)
            inst_seg = re_instance(inst_seg)

            data_id = osp.basename(self.data_infos[index]['sem_file_name']).replace(self.sem_suffix, '')

            # metric calculation post process codes:
            # 'sem_pred' has two types:
            # 1. 0-1 sem map;
            # 2. 0-1-2 sem map w/ edge;
            sem_pred = pred['sem_pred'].copy()
            dir_pred = pred['dir_pred_test']
            dir_gt = pred['dir_gt'].copy()
            if 'inst_pred' not in pred:
                if 'dir_pred' in pred:
                    dir_pred = pred['dir_pred']
                    if 'tc_sem_pred' not in pred:
                        tc_sem_pred = pred['sem_pred']
                        sem_pred = (sem_pred > 0).astype(np.uint8)
                    else:
                        tc_sem_pred = pred['tc_sem_pred']
                    # model-agnostic post process operations
                    sem_pred, inst_pred = self.model_agnostic_postprocess_w_dir(dir_pred, tc_sem_pred, sem_pred)
                elif 'tc_sem_pred' in pred:
                    tc_sem_pred = pred['tc_sem_pred']
                    sem_pred, inst_pred = self.model_agnostic_postprocess_w_tc(tc_sem_pred, sem_pred)
                else:
                    # remove edge
                    sem_pred[sem_pred == len(self.CLASSES)] = 0
                    sem_pred, inst_pred = self.model_agnostic_postprocess(sem_pred)
            else:
                sem_pred = sem_pred
                inst_pred = pred['inst_pred']

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

            sem_gt = inst_seg > 0
            inst_gt = inst_seg
            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, len(self.CLASSES))
            dir_pre_eval_res = pre_eval_all_semantic_metric(dir_pred, dir_gt, 9)

            # make contiguous ids
            inst_pred = measure.label(inst_pred.copy())
            inst_gt = measure.label(inst_gt.copy())

            inst_dice_metric = binary_inst_dice(inst_pred, inst_gt)

            pred_id_list_per_class = assign_sem_class_to_insts(inst_pred, sem_pred, len(self.CLASSES))
            gt_id_list_per_class = assign_sem_class_to_insts(inst_gt, sem_gt, len(self.CLASSES))

            # instance metric calculation
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)
            if bin_aji_pre_eval_res[0] * bin_aji_pre_eval_res[1] == 0:
                imw_aji = 0
            else:
                imw_aji = bin_aji_pre_eval_res[0] / bin_aji_pre_eval_res[1]
            aji_pre_eval_res = pre_eval_aji(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                            len(self.CLASSES))
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)
            pq_pre_eval_res = pre_eval_pq(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                          len(self.CLASSES))

            single_loop_results = dict(
                name=data_id,
                Aji=imw_aji,
                instDice=inst_dice_metric,
                Dice=dice_metric,
                Recall=recall_metric,
                Precision=precision_metric,
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                aji_pre_eval_res=aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                pq_pre_eval_res=pq_pre_eval_res,
                dir_pre_eval_res=dir_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res)
            pre_eval_results.append(single_loop_results)

            # illustrating semantic level & instance level results
            if show:
                if 'tc_sem_pred' in pred:
                    tc_sem_pred = pred['tc_sem_pred']
                else:
                    tc_sem_pred = pred['sem_pred']
                tc_sem_gt = get_tc_from_inst(inst_gt)
                draw_all(
                    show_folder,
                    img_name,
                    img_file_name,
                    sem_pred,
                    sem_gt,
                    inst_pred,
                    inst_gt,
                    tc_sem_pred,
                    tc_sem_gt,
                )

        return pre_eval_results

    def model_agnostic_postprocess(self, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_pred = (sem_pred == 1).astype(np.uint8)
        # fill instance holes
        sem_pred = binary_fill_holes(sem_pred)
        # remove small instance
        sem_pred = remove_small_objects(sem_pred, 20)
        sem_pred = sem_pred.astype(np.uint8)

        # instance process & dilation
        inst_pred = measure.label(sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))

        return sem_pred, inst_pred

    def model_agnostic_postprocess_w_tc(self, tc_sem_pred, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(sem_pred))
        sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = sem_pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 20)
            sem_canvas[sem_id_mask > 0] = sem_id
        sem_pred = sem_canvas

        # instance process & dilation
        bin_sem_pred = tc_sem_pred.copy()
        bin_sem_pred[bin_sem_pred == 2] = 0

        bin_sem_pred = binary_fill_holes(bin_sem_pred)
        bin_sem_pred = remove_small_objects(bin_sem_pred, 20)

        inst_pred = measure.label(bin_sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        # inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))
        inst_pred = align_foreground(inst_pred, sem_pred > 0, 20)

        return sem_pred, inst_pred

    def model_agnostic_postprocess_w_dir(self, dir_pred, tc_sem_pred, fore_pred):
        """model free post-process for both instance-level & semantic-level."""
        raw_fore_pred = fore_pred
        sem_pred_in = (tc_sem_pred == 1).astype(np.uint8)

        # fill instance holes
        sem_pred = binary_fill_holes(sem_pred_in)
        # remove small instance
        sem_pred = remove_small_objects(sem_pred, 20)
        sem_pred = sem_pred.astype(np.uint8)

        # instance process & dilation
        fore_pred = measure.label(sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        fore_pred = morphology.dilation(fore_pred, selem=morphology.selem.disk(2))

        raw_fore_pred = binary_fill_holes(raw_fore_pred)
        raw_fore_pred = remove_small_objects(raw_fore_pred, 20)

        sem_pred, bound = mudslide_watershed(sem_pred, dir_pred, raw_fore_pred)

        sem_pred = remove_small_objects(sem_pred, 20)
        inst_pred = measure.label(sem_pred, connectivity=1)
        inst_pred = align_foreground(inst_pred, fore_pred, 20)

        return sem_pred, inst_pred

    def evaluate(self, results, logger=None, **kwargs):
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

        img_ret_metrics = {}
        ret_metrics = {}

        img_keys = ['name', 'instDice', 'Aji', 'Dice', 'Recall', 'Precision']
        # list to dict
        for result in results:
            for key, value in result.items():
                if key in img_keys:
                    if key not in img_ret_metrics:
                        img_ret_metrics[key] = [value]
                    else:
                        img_ret_metrics[key].append(value)
                else:
                    if key not in ret_metrics:
                        ret_metrics[key] = [value]
                    else:
                        ret_metrics[key].append(value)

        # All dataset
        # semantic metrics
        sem_pre_eval_results = ret_metrics.pop('sem_pre_eval_res')
        ret_metrics.update(pre_eval_to_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))

        # dir metrics
        dir_pre_eval_results = ret_metrics.pop('dir_pre_eval_res')
        dir_metrics = pre_eval_to_sem_metrics(dir_pre_eval_results, metrics=['Dice', 'Precision', 'Recall'])

        # aji metrics
        bin_aji_pre_eval_results = ret_metrics.pop('bin_aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))
        ret_metrics.update(pre_eval_to_bin_aji(bin_aji_pre_eval_results))
        aji_pre_eval_results = ret_metrics.pop('aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_aji(aji_pre_eval_results))

        # pq metrics
        bin_pq_pre_eval_results = ret_metrics.pop('bin_pq_pre_eval_res')
        [ret_metrics.update(x) for x in pre_eval_to_bin_pq(bin_pq_pre_eval_results)]
        ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))
        pq_pre_eval_results = ret_metrics.pop('pq_pre_eval_res')
        [ret_metrics.update(x) for x in pre_eval_to_pq(pq_pre_eval_results)]
        # update per sample PQ
        [img_ret_metrics.update(x) for x in pre_eval_to_sample_pq(pq_pre_eval_results)]

        total_inst_keys = [
            'instDice', 'bAji', 'imwAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ'
        ]
        total_inst_metrics = {}
        total_analysis_keys = ['pq_bTP', 'pq_bFP', 'pq_bFN', 'pq_bIoU', 'pq_mTP', 'pq_mFP', 'pq_mFN', 'pq_mIoU']
        total_analysis = {}
        inst_analysis_keys = ['pq_TP', 'pq_FP', 'pq_FN', 'pq_IoU']
        inst_analysis = {}
        inst_keys = ['Aji', 'DQ', 'SQ', 'PQ']
        inst_metrics = {}
        sem_keys = ['Dice', 'Precision', 'Recall']
        sem_metrics = {}
        # calculate average metric
        for key in ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            if key in total_inst_keys:
                total_inst_metrics[key] = ret_metrics[key]
            elif key in inst_keys:
                # remove background class
                inst_metrics[key] = ret_metrics[key][1:]
            elif key in sem_keys:
                # remove background class
                sem_metrics[key] = ret_metrics[key][1:]
            elif key in total_analysis_keys:
                total_analysis[key] = ret_metrics[key]
            elif key in inst_analysis_keys:
                inst_analysis[key] = ret_metrics[key][1:]
            else:
                total_inst_metrics[key] = sum(ret_metrics[key]) / len(ret_metrics[key])

        # per sample
        # calculate average metric
        assert 'name' in img_ret_metrics
        name_list = img_ret_metrics.pop('name')
        name_list.append('Average')

        for key in img_ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            # if key in img_keys:
            if isinstance(img_ret_metrics[key], list):
                average_value = sum(img_ret_metrics[key]) / len(img_ret_metrics[key])
                img_ret_metrics[key].append(average_value)
                img_ret_metrics[key] = np.array(img_ret_metrics[key])
            else:
                img_ret_metrics[key] = img_ret_metrics[key].tolist()
                average_value = sum(img_ret_metrics[key]) / len(img_ret_metrics[key])
                img_ret_metrics[key].append(average_value)
                img_ret_metrics[key] = np.array(img_ret_metrics[key])

        ret_metrics_items = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in img_ret_metrics.items()
        })
        ret_metrics_items.update({'name': name_list})
        ret_metrics_items.move_to_end('name', last=False)
        items_table_data = PrettyTable()
        for key, val in ret_metrics_items.items():
            items_table_data.add_column(key, val)

        print_log('Per samples:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        total_sem_metrics = OrderedDict(
            {'m' + sem_key: np.round(np.mean(value) * 100, 2)
             for sem_key, value in sem_metrics.items()})

        # semantic table
        classes_metrics = OrderedDict()
        classes_metrics.update(
            OrderedDict({sem_key: np.round(value * 100, 2)
                         for sem_key, value in sem_metrics.items()}))
        classes_metrics.update(
            OrderedDict({inst_key: np.round(value * 100, 2)
                         for inst_key, value in inst_metrics.items()}))
        classes_metrics.update(OrderedDict({analysis_key: value for analysis_key, value in inst_analysis.items()}))

        # remove background class
        classes_metrics.update({'classes': self.CLASSES[1:]})
        classes_metrics.move_to_end('classes', last=False)

        classes_table_data = PrettyTable()
        for key, val in classes_metrics.items():
            classes_table_data.add_column(key, val)

        # total table
        total_sem_table_data = PrettyTable()
        for key, val in total_sem_metrics.items():
            total_sem_table_data.add_column(key, [val])

        total_inst_metrics = OrderedDict(
            {inst_key: np.round(value * 100, 2)
             for inst_key, value in total_inst_metrics.items()})

        total_inst_table_data = PrettyTable()
        for key, val in total_inst_metrics.items():
            total_inst_table_data.add_column(key, [val])

        total_analysis_table_data = PrettyTable()
        for key, val in total_analysis.items():
            total_analysis_table_data.add_column(key, [val])

        # direction table
        for key in dir_metrics.keys():
            # remove background class
            dir_metrics[key] = dir_metrics[key][1:]

        dir_classes_metrics = OrderedDict({dir_key: np.round(value * 100, 2) for dir_key, value in dir_metrics.items()})

        total_dir_metrics = OrderedDict(
            {'m' + dir_key: np.round(np.mean(value) * 100, 2)
             for dir_key, value in dir_metrics.items()})

        dir_classes_metrics.update({'classes': list(range(1, 9))})
        dir_classes_metrics.move_to_end('classes', last=False)

        dir_classes_table_data = PrettyTable()
        for key, val in dir_classes_metrics.items():
            dir_classes_table_data.add_column(key, val)

        dir_total_table_data = PrettyTable()
        for key, val in total_dir_metrics.items():
            dir_total_table_data.add_column(key, [val])

        print_log('Per classes:', logger)
        print_log('\n' + classes_table_data.get_string(), logger=logger)
        print_log('Semantic Total:', logger)
        print_log('\n' + total_sem_table_data.get_string(), logger=logger)
        print_log('Instance Total:', logger)
        print_log('\n' + total_inst_table_data.get_string(), logger=logger)
        print_log('Analysis Total:', logger)
        print_log('\n' + total_analysis_table_data.get_string(), logger=logger)
        print_log('Per direction classes:', logger)
        print_log('\n' + dir_classes_table_data.get_string(), logger=logger)
        print_log('Direction Total:', logger)
        print_log('\n' + dir_total_table_data.get_string(), logger=logger)

<<<<<<< HEAD
=======
        storage_results = {'total_sem_metrics': total_sem_metrics, 'total_inst_metrics': total_inst_metrics, 'class_inst_metrics': classes_metrics}

>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
        eval_results = {}
        # average results
        if 'Aji' in img_ret_metrics:
            eval_results['Aji'] = img_ret_metrics['Aji'][-1]
        if 'Dice' in img_ret_metrics:
            eval_results['Dice'] = img_ret_metrics['Dice'][-1]
        if 'Recall' in img_ret_metrics:
            eval_results['Recall'] = img_ret_metrics['Recall'][-1]
        if 'Precision' in img_ret_metrics:
            eval_results['Precision'] = img_ret_metrics['Precision'][-1]

        for k, v in total_sem_metrics.items():
            eval_results[k] = v
        for k, v in total_inst_metrics.items():
            eval_results[k] = v

        classes = classes_metrics.pop('classes', None)
        for key, value in classes_metrics.items():
            eval_results.update({key + '.' + str(name): f'{value[idx]:.3f}' for idx, name in enumerate(classes)})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
<<<<<<< HEAD
        return eval_results
=======
        return eval_results, storage_results
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
