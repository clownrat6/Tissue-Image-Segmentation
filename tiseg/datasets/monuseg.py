import os
import os.path as osp
import warnings
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology
from skimage.morphology import remove_small_objects
from torch.utils.data import Dataset

from tiseg.utils.evaluation.metrics import (aggregated_jaccard_index,
                                            binary_dice_similarity_coefficient,
                                            binary_precision_recall)
from .builder import DATASETS
from .pipelines import Compose


def draw_semantic(save_folder, data_id, image, pred, label,
                  single_loop_results):
    """draw semantic level picture with FP & FN."""
    import matplotlib.pyplot as plt

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
    canvas[label == 1, :] = (255, 255, 2)
    canvas[(pred == 0) * (label == 1), :] = (2, 255, 255)
    canvas[(pred == 1) * (label == 0), :] = (255, 2, 255)
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
    aji = f'{single_loop_results["aji"] * 100:.2f}'
    dice = f'{single_loop_results["dice"] * 100:.2f}'
    recall = f'{single_loop_results["recall"] * 100:.2f}'
    precision = f'{single_loop_results["precision"] * 100:.2f}'
    temp_str = (f'aji: {aji:<10}\ndice: '
                f'{dice:<10}\nrecall: {recall:<10}\nprecision: '
                f'{precision:<10}')
    plt.suptitle(temp_str, fontsize=15, color='black')
    plt.tight_layout()
    plt.savefig(
        f'{save_folder}/{data_id}_monuseg_semantic_compare.png', dpi=200)


def draw_instance(save_folder, data_id, pred_instance, label_instance):
    """draw instance level picture."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5 * 2, 5))

    plt.subplot(121)
    plt.imshow(pred_instance)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(label_instance)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(
        f'{save_folder}/{data_id}_monuseg_instance_compare.png', dpi=200)


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
                 ann_suffix='_instance.npy',
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
                                                self.ann_suffix, self.split)

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
                    img_id = line.strip()
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
            seg_map = osp.join(self.ann_dir, data_info['ann_name'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            yield gt_seg_map

    def pre_eval(self,
                 preds,
                 indices,
                 show_semantic=False,
                 show_instance=False,
                 show_folder=None):
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
            warnings.warn(
                'show_semantic or show_instance is set to True, but the '
                'show_folder is None. We will use default show_folder: '
                './monuseg_show')
            show_folder = '.monuseg_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir,
                               self.data_infos[index]['ann_name'])
            seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')
            data_id = self.data_infos[index]['ann_name'].split('_')[0]

            # metric calculation post process codes:
            # extract inside
            pred_edge = (pred == 2).astype(np.uint8)
            seg_map_edge = (seg_map == 2).astype(np.uint8)
            pred = (pred == 1).astype(np.uint8)
            seg_map = (seg_map == 1).astype(np.uint8)

            # model-agnostic post process operations
            pred_semantic, pred_instance = self.model_agnostic_postprocess(
                pred)
            seg_map_semantic, seg_map_instance = seg_map.copy(), seg_map.copy()

            # TODO: (Important issue about post process)
            # This may be the dice metric calculation trick (Need be
            # considering carefully)
            # convert instance map (after postprocess) to semantic level
            pred_semantic = (pred_instance > 0).astype(np.uint8)
            seg_map_semantic = (seg_map_instance > 0).astype(np.uint8)

            # semantic metric calculation
            precision_metric, recall_metric = binary_precision_recall(
                pred_semantic, seg_map_semantic)
            dice_metric = binary_dice_similarity_coefficient(
                pred_semantic, seg_map_semantic)
            edge_precision_metric, edge_recall_metric = \
                binary_precision_recall(pred_edge, seg_map_edge)
            edge_dice_metric = binary_dice_similarity_coefficient(
                pred_edge, seg_map_edge)

            # instance metric calculation
            seg_map_instance = measure.label(seg_map_instance)
            aji_metric = aggregated_jaccard_index(
                pred_instance, seg_map_instance, is_semantic=False)

            single_loop_results = dict(
                name=data_id,
                aji=aji_metric,
                dice=dice_metric,
                recall=recall_metric,
                precision=precision_metric,
                edge_dice=edge_dice_metric,
                edge_recall=edge_recall_metric,
                edge_precision=edge_precision_metric)
            pre_eval_results.append(single_loop_results)

            # illustrating semantic level results
            if show_semantic:
                data_info = self.data_infos[index]
                image_path = osp.join(self.img_dir, data_info['img_name'])
                draw_semantic(show_folder, data_id, image_path, pred_semantic,
                              seg_map_semantic, single_loop_results)

            # illustrating instance level results
            if show_instance:
                draw_instance(show_folder, data_id, pred_instance,
                              seg_map_instance)

        return pre_eval_results

    def model_agnostic_postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        # fill instance holes
        pred = binary_fill_holes(pred)
        # remove small instance
        pred = remove_small_objects(pred, 20)
        pred = pred.astype(np.uint8)
        pred_semantic = pred.copy()

        # instance process & dilation
        pred = pred.copy()
        pred_instance = measure.label(pred)
        # if re_edge=True, dilation pixel length should be 2
        pred_instance = morphology.dilation(
            pred_instance, selem=morphology.disk(1))

        return pred_semantic, pred_instance

    def evaluate(self,
                 results,
                 metric='all',
                 logger=None,
                 dump_path=None,
                 **kwargs):
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
        allowed_metrics = ['all']
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
            average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
            ret_metrics[key].append(average_value)
            ret_metrics[key] = np.array(ret_metrics[key])

        # TODO: Refactor for more general metric
        # if dump_path is not None:
        #     fp = open(f'{dump_path}', 'w')
        #     head_info = f'{"item_name":<30} | '
        #     key_list = ret_metrics.keys()
        #     # make metric record head info
        #     for key in key_list:
        #         head_info += f'{key:<30} | '
        #     fp.write(head_info + '\n')
        #     for idx, name in enumerate(name_list):
        #         # make single line info
        #         single_line_info = f'{name:<30} | '
        #         for key in key_list:
        #             format_value = f'{ret_metrics[key][idx] * 100:.2f}'
        #             single_line_info += f'{format_value:<30} | '
        #         fp.write(single_line_info + '\n')

        # for logger
        ret_metrics_items = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_items.update({'name': name_list})
        ret_metrics_items.move_to_end('name', last=False)
        items_table_data = PrettyTable()
        for key, val in ret_metrics_items.items():
            items_table_data.add_column(key, val)

        print_log('Per class:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        # dump to txt
        if dump_path is not None:
            fp = open(f'{dump_path}', 'w')
            fp.write(items_table_data.get_string())

        eval_results = {}
        # average results
        if 'aji' in ret_metrics:
            eval_results['aji'] = ret_metrics['aji'][-1]
        if 'dice' in ret_metrics:
            eval_results['dice'] = ret_metrics['dice'][-1]
        if 'recall' in ret_metrics:
            eval_results['recall'] = ret_metrics['recall'][-1]
        if 'precision' in ret_metrics:
            eval_results['precision'] = ret_metrics['precision'][-1]
        if 'edge_dice' in ret_metrics:
            eval_results['edge_dice'] = ret_metrics['edge_dice'][-1]
        if 'edge_recall' in ret_metrics:
            eval_results['edge_recall'] = ret_metrics['edge_recall'][-1]
        if 'edge_precision' in ret_metrics:
            eval_results['edge_precision'] = ret_metrics['edge_precision'][-1]

        ret_metrics_items.pop('name', None)
        for key, value in ret_metrics_items.items():
            eval_results.update({
                key + '.' + str(name): f'{value[idx]:.3f}'
                for idx, name in enumerate(name_list)
            })

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
