import os
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from tiseg.utils import (aggregated_jaccard_index, dice_similarity_coefficient, precision_recall)
from .builder import DATASETS
from .custom import CustomDataset
from .utils import re_instance


@DATASETS.register_module()
class OSCDDataset(CustomDataset):
    """OSCD dataset is similar to two-classes nuclei segmentation dataset."""

    CLASSES = ('background', 'carton')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', sem_suffix='_sem.png', inst_suffix='_inst.npy', **kwargs)

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
            precision_metric, recall_metric = precision_recall(sem_pred, sem_gt, 2)
            precision_metric = precision_metric[1]
            recall_metric = recall_metric[1]
            dice_metric = dice_similarity_coefficient(sem_pred, sem_gt, 2)[1]

            # instance metric calculation
            aji_metric = aggregated_jaccard_index(re_instance(inst_pred), inst_gt)

            single_loop_results = dict(
                Aji=aji_metric, Dice=dice_metric, Recall=recall_metric, Precision=precision_metric)
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
        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        inst_eval = ['Aji']
        sem_eval = ['IoU', 'Dice', 'Precision', 'Recall']
        inst_metrics = {}
        sem_metrics = {}
        # calculate average metric
        for key in ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
            if key in inst_eval:
                inst_metrics[key] = average_value
            elif key in sem_eval:
                sem_metrics[key] = average_value

        # total table
        total_metrics = OrderedDict()
        sem_total_metrics = OrderedDict(
            {'m' + sem_key: np.round(value * 100, 2)
             for sem_key, value in sem_metrics.items()})
        inst_total_metrics = OrderedDict(
            {inst_key: np.round(value * 100, 2)
             for inst_key, value in inst_metrics.items()})

        total_metrics.update(sem_total_metrics)
        total_metrics.update(inst_total_metrics)

        total_table_data = PrettyTable()
        for key, val in total_metrics.items():
            total_table_data.add_column(key, [val])

        print_log('Total:', logger)
        print_log('\n' + total_table_data.get_string(), logger=logger)

        eval_results = {}
        # average results
        for k, v in total_metrics.items():
            eval_results[k] = v

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
