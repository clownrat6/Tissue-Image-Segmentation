import warnings
import os
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from tiseg.utils import (pre_eval_all_semantic_metric, pre_eval_to_sem_metrics, pre_eval_bin_aji, pre_eval_bin_pq,
                         pre_eval_to_aji, pre_eval_to_pq, pre_eval_to_inst_dice, pre_eval_to_imw_pq,
                         pre_eval_to_imw_aji, pre_eval_to_imw_inst_dice, pre_eval_to_imw_sem_metrics,
                         pre_eval_to_bin_aji, pre_eval_to_bin_pq)
from .builder import DATASETS
from .utils import re_instance
from .custom import CustomDataset


@DATASETS.register_module()
class MoNuSegDatasetDebug(CustomDataset):
    """MoNuSeg Nuclei Segmentation Dataset.

    MoNuSeg is actually instance segmentation task dataset. However, it can be
    seen as a two class semantic segmentation task (Background, Nuclei).
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.tif', sem_suffix='_sem.png', inst_suffix='_inst.npy', **kwargs)

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
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_gt = re_instance(np.load(inst_file_name))

            data_id = osp.basename(self.data_infos[index]['sem_file_name']).replace(self.sem_suffix, '')

            # metric calculation post process codes:
            # 'sem_pred' is a three-class map: 0-1-2 sem map w/ edge;
            sem_pred = pred['sem_pred']
            inst_pred = pred['inst_pred']
            tc_pred = pred['tc_pred']
            tc_gt = pred['tc_gt']

            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, len(self.CLASSES))
            bound_sem_pre_eval_res = pre_eval_all_semantic_metric(tc_pred, tc_gt, len(self.CLASSES) + 1)

            # make contiguous ids
            inst_pred = re_instance(inst_pred)
            inst_gt = re_instance(inst_gt)

            # instance metric calculation
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)

            single_loop_results = dict(
                name=data_id,
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                bound_sem_pre_eval_res=bound_sem_pre_eval_res,
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
        Returns:
            dict[str, float]: Default metrics.
        """

        img_ret_metrics = {}
        ret_metrics = {}

        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # All dataset
        # name id
        img_ret_metrics['name'] = ret_metrics.pop('name')

        # bound metrics
        bound_sem_pre_eval_results = ret_metrics.pop('bound_sem_pre_eval_res')
        for k, v in pre_eval_to_sem_metrics(bound_sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']).items():
            ret_metrics['Bound' + k] = v[-1]

        # semantic metrics
        sem_pre_eval_results = ret_metrics.pop('sem_pre_eval_res')
        ret_metrics.update(pre_eval_to_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))
        img_ret_metrics.update(
            pre_eval_to_imw_sem_metrics(sem_pre_eval_results, metrics=['Dice', 'Precision', 'Recall']))

        # instance metrics (aji style)
        bin_aji_pre_eval_results = ret_metrics.pop('bin_aji_pre_eval_res')
        ret_metrics.update(pre_eval_to_aji(bin_aji_pre_eval_results))
        for k, v in pre_eval_to_bin_aji(bin_aji_pre_eval_results).items():
            ret_metrics['b' + k] = v
        img_ret_metrics.update(pre_eval_to_imw_aji(bin_aji_pre_eval_results))

        # instance metrics (pq style)
        bin_pq_pre_eval_results = ret_metrics.pop('bin_pq_pre_eval_res')
        ret_metrics.update(pre_eval_to_pq(bin_pq_pre_eval_results))
        for k, v in pre_eval_to_bin_pq(bin_pq_pre_eval_results).items():
            ret_metrics['b' + k] = v
        ret_metrics.update(pre_eval_to_inst_dice(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_pq(bin_pq_pre_eval_results))
        img_ret_metrics.update(pre_eval_to_imw_inst_dice(bin_pq_pre_eval_results))

        assert 'name' in img_ret_metrics
        name_list = img_ret_metrics.pop('name')
        name_list.append('Average')

        # insert average value into image wise list
        for key in img_ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            if len(img_ret_metrics[key].shape) == 2:
                img_ret_metrics[key] = img_ret_metrics[key][:, 0]
            average_value = np.nanmean(img_ret_metrics[key])
            img_ret_metrics[key] = img_ret_metrics[key].tolist()
            img_ret_metrics[key].append(average_value)
            img_ret_metrics[key] = np.array(img_ret_metrics[key])

        vital_keys = ['Dice', 'Precision', 'Recall', 'Aji', 'DQ', 'SQ', 'PQ', 'InstDice']
        mean_metrics = {}
        overall_metrics = {}
        # calculate average metric
        for key in vital_keys:
            # XXX: Using average value may have lower metric value than using
            mean_metrics['imw' + key] = img_ret_metrics[key][-1]
            overall_metrics['m' + key] = ret_metrics[key]

        for key in ['bAji', 'bDQ', 'bSQ', 'bPQ']:
            overall_metrics[key] = ret_metrics[key]

        for key in ['BoundDice', 'BoundPrecision', 'BoundRecall']:
            overall_metrics[key] = ret_metrics[key]

        # per sample
        sample_metrics = OrderedDict(
            {sample_key: np.round(metric_value * 100, 2)
             for sample_key, metric_value in img_ret_metrics.items()})
        sample_metrics.update({'name': name_list})
        sample_metrics.move_to_end('name', last=False)

        items_table_data = PrettyTable()
        for key, val in sample_metrics.items():
            items_table_data.add_column(key, val)

        print_log('Per samples:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        # mean table
        mean_metrics = OrderedDict(
            {mean_key: np.round(np.mean(value) * 100, 2)
             for mean_key, value in mean_metrics.items()})

        mean_table_data = PrettyTable()
        for key, val in mean_metrics.items():
            mean_table_data.add_column(key, [val])

        # overall table
        overall_metrics = OrderedDict(
            {sem_key: np.round(np.mean(value) * 100, 2)
             for sem_key, value in overall_metrics.items()})

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

        for k, v in mean_metrics.items():
            eval_results[k] = v
        for k, v in overall_metrics.items():
            eval_results[k] = v

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results, storage_results
