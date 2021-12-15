import os
import os.path as osp
import warnings
from collections import OrderedDict

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
from .utils import re_instance


@DATASETS.register_module()
class NucleiCoNICDataset(Dataset):
    """Nuclei Custom Foundation Segmentation Dataset.

    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, Nuclei1, Nuclei2, ...).
    The basic settings only supports two-class nuclei segmentation task.

    related suffix:
        "_semantic.png": raw semantic map (seven class semantic map without
            boundary).
        "_instance.npy": instance level map.
    """

    CLASSES = ('background', 'neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self,
                 process_cfg,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 img_suffix='.png',
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
            sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_gt = np.load(inst_file_name)
            inst_gt = re_instance(inst_gt)

            # metric calculation post process codes:
            sem_pred = pred
            # remove edge
            sem_pred[sem_pred == (len(self.CLASSES) - 1)] = 0

            # model-agnostic post process operations
            sem_pred, inst_pred = self.model_agnostic_postprocess(sem_pred)

            # semantic metric calculation (remove background class)
            # [1] will remove background class.
            precision_metric, recall_metric = precision_recall(sem_pred, sem_gt, len(self.CLASSES))[1:]
            dice_metric = dice_similarity_coefficient(sem_pred, sem_gt, 2)[1:]

            # instance metric calculation
            aji_metric = aggregated_jaccard_index(inst_pred, inst_gt, is_semantic=False)

            single_loop_results = dict(
                Aji=aji_metric, Dice=dice_metric, Recall=recall_metric, Precision=precision_metric)
            pre_eval_results.append(single_loop_results)

        return pre_eval_results

    def model_agnostic_postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(pred))
        inst_pred = np.zeros_like(pred).astype(np.uint8)
        sem_pred = np.zeros_like(pred).astype(np.uint8)
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            # remove small instance
            sem_id_mask = remove_small_objects(sem_id_mask, 20)
            sem_id_mask_dila = morphology.dilation(sem_id_mask, selem=morphology.disk(2))
            inst_pred[sem_id_mask > 0] = 1
            sem_pred[sem_id_mask_dila > 0] = sem_id

        # instance process & dilation
        inst_pred = measure.label(inst_pred)
        # if re_edge=True, dilation pixel length should be 2
        inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))

        return sem_pred, inst_pred

    def evaluate(self, results, metric='all', logger=None, dump_path=None, **kwargs):
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

        # dump to txt
        if dump_path is not None:
            fp = open(f'{dump_path}', 'w')
            fp.write(items_table_data.get_string())

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