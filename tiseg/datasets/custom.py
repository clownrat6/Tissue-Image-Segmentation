import json
import os
import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from tiseg.utils.evaluation.metrics import (eval_metrics, intersect_and_union,
                                            pre_eval_to_metrics)
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class CustomDataset(Dataset):

    vocab = None

    CLASSES = ('background', 'referring object')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self,
                 pipelines,
                 img_dir,
                 ann_dir,
                 sent_dir,
                 data_root=None,
                 img_suffix='.jpg',
                 ann_suffix='.png',
                 sent_suffix='.json',
                 test_mode=False,
                 split=None):

        self.pipelines = Compose(pipelines)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.sent_dir = sent_dir
        self.data_root = data_root

        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix
        self.sent_suffix = sent_suffix

        self.test_mode = test_mode
        self.split = split

        assert sent_suffix == '.json', 'It is only support json '
        'sentence annotation now.'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.sent_dir is None or osp.isabs(self.sent_dir)):
                self.sent_dir = osp.join(self.data_root, self.sent_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.data_infos = self.load_annotations(self.sent_dir, self.img_suffix,
                                                self.ann_suffix,
                                                self.sent_suffix, self.split)

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
        if self.test_mode:
            return self.prepare_test_data(index)
        else:
            return self.prepare_train_data(index)

    def prepare_test_data(self, index):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        data_info = self.data_infos[index]
        results = self.pre_pipeline(data_info)
        return self.pipelines(results)

    def prepare_train_data(self, index):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        data_info = self.data_infos[index]
        results = self.pre_pipeline(data_info)
        return self.pipelines(results)

    def pre_pipeline(self, data_info):
        """Prepare results dict for pipeline."""
        results = {}
        results['img_info'] = {}
        results['ann_info'] = {}
        results['txt_info'] = {}
        results['sent_info'] = {}

        # path retrieval
        results['img_info']['img_name'] = data_info['img_name']
        results['img_info']['img_dir'] = self.img_dir
        results['ann_info']['ann_name'] = data_info['ann_name']
        results['ann_info']['ann_dir'] = self.ann_dir
        results['sent_info']['sent_name'] = data_info['sent_name']
        results['sent_info']['sent_dir'] = self.sent_dir

        # build seg fileds
        results['seg_fields'] = []

        return results

    def load_annotations(self, sent_dir, img_suffix, ann_suffix, sent_suffix,
                         split):
        """Load annotation from directory.

        In this task, we set one sent, sent related image and seg map as
        a batch.

        Args:
            sent (str): Path to sent info directory
            img_suffix (str): Suffix of images.
            ann_suffix (str): Suffix of segmentation maps.
            sent_suffix (str): Suffix of sent info.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all sents
                in sent_dir will be loaded. Default: None

        Returns:
            list[dict]: All data info of dataset, data info contains image,
                segmentation map and sent info filename.
        """
        data_infos = []
        if split is not None:
            with open(split, 'r') as fp:
                for line in fp.readlines():
                    img_id, ann_id, sent_id = line.strip().split()
                    sent_name = sent_id + sent_suffix
                    image_name = img_id + img_suffix
                    ann_name = ann_id + ann_suffix
                    data_info = dict(
                        img_name=image_name,
                        ann_name=ann_name,
                        sent_name=sent_name)
                    data_infos.append(data_info)
        else:
            for sent_name in os.listdir(self.sent_dir):
                sent_path = osp.join(sent_dir, sent_name)
                sent_info = json.load(open(sent_path, 'r'))
                data_info = dict(
                    img_name=sent_info['image_name'],
                    ann_name=sent_info['ann_name'],
                    sent_name=sent_name)
                data_infos.append(data_info)

        return data_infos

    def get_gt_instance_maps(self):
        """Ground Truth maps generator."""
        for data_info in self.data_infos:
            instance_map = osp.join(self.ann_dir, data_info['ann_name'])
            gt_instance_map = mmcv.imread(
                instance_map, flag='unchanged', backend='pillow')
            yield gt_instance_map

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir,
                               self.data_infos[index]['ann_name'])
            seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            pre_eval_results.append(
                intersect_and_union(pred, seg_map, len(self.CLASSES)))

        return pre_eval_results

    def format_results(self, results, **kwargs):
        """Save prediction results for submit."""
        pass

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, str):
            gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            # reset generator
            gt_seg_maps = self.get_gt_seg_maps()
            ret_metrics = eval_metrics(results, gt_seg_maps, num_classes,
                                       metric)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
