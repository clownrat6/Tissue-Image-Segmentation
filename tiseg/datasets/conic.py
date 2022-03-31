import os
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from tiseg.utils import (pre_eval_all_semantic_metric, pre_eval_bin_aji, pre_eval_bin_pq, pre_eval_to_sem_metrics,
                         pre_eval_to_imw_sem_metrics, pre_eval_aji, pre_eval_pq, pre_eval_to_bin_aji, pre_eval_to_aji,
                         pre_eval_to_bin_pq, pre_eval_to_pq, pre_eval_to_imw_pq, pre_eval_to_imw_aji)
from .builder import DATASETS
from .dataset_mapper import DatasetMapper
from .utils import re_instance, assign_sem_class_to_insts


@DATASETS.register_module()
class CoNICDataset(Dataset):
    """Nuclei Custom Foundation Segmentation Dataset.
    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, Nuclei1, Nuclei2, ...).
    The basic settings only supports two-class nuclei segmentation task.
    related suffix:
        "_sem.png": raw semantic map (seven class semantic map without
            boundary).
        "_inst.npy": instance level map.
    """

    CLASSES = ('background', 'neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

    def __init__(self,
                 processes,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 img_suffix='.png',
                 sem_suffix='_sem.png',
                 inst_suffix='_inst.npy',
                 test_mode=False,
                 split=None):

        self.mapper = DatasetMapper(test_mode, processes=processes)

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

    def pre_eval(self, preds, indices, show=False, show_folder='.nuclei_show'):
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
            warnings.warn('show is set to True, but the '
                          'show_folder is None. We will use default show_folder: '
                          '.nuclei_show')
            show_folder = '.nuclei_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            # img related infos
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_gt = np.load(inst_file_name)
            inst_gt = re_instance(inst_gt)

            # metric calculation & post process codes:
            sem_pred = pred['sem_pred'].copy()
            inst_pred = pred['inst_pred'].copy()

            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, len(self.CLASSES))

            # make contiguous ids
            inst_pred = re_instance(inst_pred)
            inst_gt = re_instance(inst_gt)

            pred_id_list_per_class = assign_sem_class_to_insts(inst_pred, sem_pred, len(self.CLASSES))
            gt_id_list_per_class = assign_sem_class_to_insts(inst_gt, sem_gt, len(self.CLASSES))

            # instance metric calculation
            aji_pre_eval_res = pre_eval_aji(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                            len(self.CLASSES))
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)

            pq_pre_eval_res = pre_eval_pq(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                          len(self.CLASSES))
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)

            single_loop_results = dict(
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                aji_pre_eval_res=aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                pq_pre_eval_res=pq_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res)
            pre_eval_results.append(single_loop_results)

        return pre_eval_results

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'Aji',
                'Dice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            dump_path (str | None, optional): The dump path of each item
                evaluation results. Default: None
        Returns:
            dict[str, float]: Default metrics.
        """

        ret_metrics = {}
        img_ret_metrics = {}

        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # semantic metrics
        sem_pre_eval_results = ret_metrics.pop('sem_pre_eval_res')
        ret_metrics.update(pre_eval_to_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))
        img_ret_metrics.update(
            pre_eval_to_imw_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))

        # aji metrics
        aji_pre_eval_results = ret_metrics.pop('aji_pre_eval_res')
        bin_aji_pre_eval_results = ret_metrics.pop('bin_aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_aji(aji_pre_eval_results))
        for k, v in pre_eval_to_bin_aji(bin_aji_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))

        # pq metrics
        pq_pre_eval_results = ret_metrics.pop('pq_pre_eval_res')
        bin_pq_pre_eval_results = ret_metrics.pop('bin_pq_pre_eval_res')
        ret_metrics.update(pre_eval_to_pq(pq_pre_eval_results))
        for k, v in pre_eval_to_bin_pq(bin_pq_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))

        vital_keys = ['Dice', 'Precision', 'Recall', 'Aji', 'DQ', 'SQ', 'PQ']
        mean_metrics = {}
        overall_metrics = {}
        classes_metrics = OrderedDict()
        # calculate average metric
        for key in vital_keys:
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            mean_metrics['imw' + key] = np.nanmean(img_ret_metrics[key])
            overall_metrics['m' + key] = np.nanmean(ret_metrics[key])
            # class wise metric
            classes_metrics[key] = ret_metrics[key]
            average_value = np.nanmean(classes_metrics[key])
            tmp_list = classes_metrics[key].tolist()
            tmp_list.append(average_value)
            classes_metrics[key] = np.array(tmp_list)

        for key in ['bAji', 'bDQ', 'bSQ', 'bPQ']:
            overall_metrics[key] = ret_metrics[key]

        # class wise table
        classes_metrics.update(
            OrderedDict({class_key: np.round(value * 100, 2)
                         for class_key, value in classes_metrics.items()}))
        # classes_metrics.update(OrderedDict({analysis_key: value for analysis_key, value in inst_analysis.items()}))

        # remove background class
        classes_metrics.update({'classes': list(self.CLASSES[1:]) + ['average']})
        classes_metrics.move_to_end('classes', last=False)

        classes_table_data = PrettyTable()
        for key, val in classes_metrics.items():
            classes_table_data.add_column(key, val)

        print_log('Per classes:', logger)
        print_log('\n' + classes_table_data.get_string(), logger=logger)

        # mean table
        mean_metrics = OrderedDict({key: np.round(np.mean(value) * 100, 2) for key, value in mean_metrics.items()})

        mean_table_data = PrettyTable()
        for key, val in mean_metrics.items():
            mean_table_data.add_column(key, [val])

        # overall table
        overall_metrics = OrderedDict(
            {key: np.round(np.mean(value) * 100, 2)
             for key, value in overall_metrics.items()})

        overall_table_data = PrettyTable()
        for key, val in overall_metrics.items():
            overall_table_data.add_column(key, [val])

        print_log('Mean Total:', logger)
        print_log('\n' + mean_table_data.get_string(), logger=logger)
        print_log('Overall Total:', logger)
        print_log('\n' + overall_table_data.get_string(), logger=logger)

        storage_results = {
            'mean_metrics': mean_metrics,
            'overall_metrics': overall_metrics,
        }

        eval_results = {}
        for k, v in overall_metrics.items():
            eval_results[k] = v
        for k, v in mean_metrics.items():
            eval_results[k] = v

        classes = classes_metrics.pop('classes', None)
        for key, value in classes_metrics.items():
            eval_results.update({key + '.' + str(name): f'{value[idx]:.3f}' for idx, name in enumerate(classes)})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results, storage_results
