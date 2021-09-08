import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from tiseg.utils.evaluation.metrics import (aggregated_jaccard_index,
                                            intersect_and_union)
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class MoNuSegDataset(Dataset):
    """MoNuSeg Nuclei Segmentation Dataset.

    MoNuSeg is actually instance segmentation task dataset. However, it can
    seem as a three class semantic segmentation task (Background, Nuclei, Edge)
    """

    CLASSES = ('background', 'nuclei', 'edge')

    PALETTE = [[0, 0, 0], [255, 2, 255], [2, 255, 255]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 img_suffix='.tif',
                 ann_suffix='_semantic_with_edge.png',
                 test_mode=False,
                 split=None):

        self.pipeline = Compose(pipeline)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.data_root = data_root

        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix

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

        self.data_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                                self.ann_suffix)

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
        return self.pipeline(results)

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
        return self.pipeline(results)

    def pre_pipeline(self, data_info):
        """Prepare results dict for pipeline."""
        results = {}
        results['img_info'] = {}
        results['ann_info'] = {}

        # path retrieval
        results['img_info']['img_name'] = data_info['img_name']
        results['img_info']['img_dir'] = self.img_dir
        results['ann_info']['ann_name'] = data_info['ann_name']
        results['ann_info']['ann_dir'] = self.ann_dir

        # build seg fileds
        results['seg_fields'] = []

        return results

    def load_annotations(self, img_dir, img_suffix, ann_suffix, split=None):
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
                    img_id = line.strip().split()
                    image_name = img_id + img_suffix
                    ann_name = img_id + ann_suffix
                    data_info = dict(img_name=image_name, ann_name=ann_name)
                    data_infos.append(data_info)
        else:
            for img_name in mmcv.scandir(img_dir, img_suffix, recursive=True):
                ann_name = img_name.replace(img_suffix, ann_suffix)
                data_info = dict(img_name=img_name, ann_name=ann_name)
                data_infos.append(data_info)

        return data_infos

    def get_gt_seg_maps(self):
        """Ground Truth maps generator."""
        for data_info in self.data_infos:
            instance_map = osp.join(self.ann_dir, data_info['ann_name'])
            gt_seg_map = mmcv.imread(
                instance_map, flag='unchanged', backend='pillow')
            yield gt_seg_map

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
            # pre eval aji and dice metric
            aji_metric = aggregated_jaccard_index(pred, seg_map)

            intersect, union, _, _ = intersect_and_union(pred, seg_map, 2)
            dice_metric = (2 * intersect / (union + intersect))[1].numpy()

            pre_eval_results.append(dict(aji=aji_metric, dice=dice_metric))

        return pre_eval_results

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
        allowed_metrics = ['aji', 'dice', 'all']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        ret_metrics = {}
        # test a list of files
        if 'all' in metric:
            # dict to list
            aji_list = []
            dice_list = []
            for item in results:
                aji_list.append(item['aji'])
                dice_list.append(item['dice'])
            eval_results['aji'] = ret_metrics['aji'] = np.array(
                [sum(aji_list) / len(aji_list)])
            eval_results['dice'] = ret_metrics['dice'] = np.array(
                [sum(dice_list) / len(dice_list)])

        if 'aji' in metric:
            aji_list = []
            for item in results:
                aji_list.append(item['aji'])
            ret_metrics['aji'] = np.array([sum(aji_list) / len(aji_list)])
            eval_results['aji'] = ret_metrics['aji']
        if 'dice' in metric:
            dice_list = []
            for item in results:
                dice_list.append(item['dice'])
            ret_metrics['dice'] = np.array([sum(dice_list) / len(dice_list)])
            eval_results['dice'] = ret_metrics['dice']

        # for logger
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': ['Nuclei']})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('Per class:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(['Nuclei'])
            })

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
