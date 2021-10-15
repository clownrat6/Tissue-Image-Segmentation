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

from tiseg.utils.evaluation.metrics import (aggregated_jaccard_index,
                                            pre_eval_all_semantic_metric,
                                            pre_eval_to_metrics)
from .builder import DATASETS
from .pipelines import Compose
from .utils import draw_instance, draw_semantic, re_instance


@DATASETS.register_module()
class InstanceCOCOStuffDataset(Dataset):
    """Semantic-to-Instance Segmentation Dataset (with stuff).

    Although, this dataset is a instance segmentation task, this dataset also
    support a multiple class semantic segmentation task (Background, xxx, yyy,
    zzz, ...,Edge).

    When evaluation, we will remove stuff prediction.

    related suffix:
        "_semantic.png": multi class semantic map.
        "_semantic_with_edge.png": multi class semantic map with edge.
        "_instance.npy": instance level map.
    """

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    EDGE_ID = 171

    PALETTE = [(147, 128, 82), (17, 164, 238), (65, 110, 148), (141, 242, 140),
               (115, 200, 43), (93, 36, 213), (152, 77, 133), (178, 97, 232),
               (162, 124, 200), (185, 124, 96), (45, 76, 240), (120, 11, 205),
               (86, 6, 120), (137, 45, 205), (35, 228, 99), (159, 117, 153),
               (144, 103, 53), (28, 195, 246), (159, 189, 231), (190, 141, 44),
               (37, 216, 79), (40, 41, 5), (0, 31, 26), (218, 175, 206),
               (92, 2, 171), (87, 23, 131), (28, 77, 83), (139, 202, 212),
               (146, 106, 124), (179, 50, 185), (146, 168, 113),
               (11, 144, 219), (52, 150, 114), (71, 129, 242), (216, 92, 253),
               (5, 51, 78), (169, 173, 144), (130, 198, 171), (62, 191, 243),
               (191, 115, 71), (218, 221, 139), (96, 30, 43), (167, 122, 157),
               (109, 159, 85), (52, 141, 239), (130, 228, 169), (12, 96, 36),
               (245, 163, 163), (79, 42, 211), (132, 40, 96), (162, 180, 236),
               (92, 155, 191), (110, 48, 115), (64, 218, 144), (77, 230, 229),
               (184, 24, 51), (166, 54, 133), (48, 73, 227), (146, 112, 208),
               (35, 105, 187), (167, 181, 141), (77, 185, 60), (48, 45, 237),
               (196, 90, 229), (231, 87, 212), (75, 139, 72), (119, 194, 95),
               (185, 188, 169), (244, 188, 151), (177, 239, 76), (96, 224, 76),
               (151, 67, 174),
               (250, 210, 237), (220, 221, 179), (251, 147, 133),
               (131, 185, 120), (153, 99, 11), (169, 197, 249), (197, 62, 20),
               (4, 7, 184)]

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir,
                 ignore_index=255,
                 data_root=None,
                 img_suffix='.jpg',
                 ann_suffix='_semantic_with_edge.png',
                 test_mode=False,
                 split=None):

        self.pipeline = Compose(pipeline)

        self.ignore_index = ignore_index

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
        results['img_info']['img_suffix'] = self.img_suffix
        results['ann_info']['ann_name'] = data_info['ann_name']
        results['ann_info']['ann_dir'] = self.ann_dir
        results['ann_info']['ann_suffix'] = self.ann_suffix

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
                '.coco_stuff_show')
            show_folder = '.coco_stuff_show'
            if not osp.exists(show_folder):
                os.makedirs(show_folder, 0o775)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir,
                               self.data_infos[index]['ann_name'])
            # semantic level label make (with edge)
            seg_map_semantic_edge = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            seg_map_edge = (seg_map_semantic_edge == self.EDGE_ID).astype(
                np.uint8)
            # ground truth of semantic level
            seg_map_semantic = seg_map.replace('_semantic_with_edge.png',
                                               '_semantic.png')
            seg_map_semantic = mmcv.imread(
                seg_map_semantic, flag='unchanged', backend='pillow')
            # instance level label make
            seg_map_instance = seg_map.replace('_semantic_with_edge.png',
                                               '_instance.npy')
            seg_map_instance = np.load(seg_map_instance)
            seg_map_instance = re_instance(seg_map_instance)

            # metric calculation post process codes:
            # extract semantic results w/ edge
            pred_semantic_edge = pred
            # edge id is 171 (len(self.CLASSES) - 1)
            pred_edge = (pred_semantic_edge == self.EDGE_ID).astype(np.uint8)
            pred_semantic = pred.copy()
            pred_semantic[pred_edge > 0] = 0

            # model-agnostic post process operations
            pred_semantic, pred_instance = self.model_agnostic_postprocess(
                pred_semantic)

            # semantic metric calculation (remove background class)
            # [1] will remove background class.
            semantic_pre_eval_results = pre_eval_all_semantic_metric(
                pred_semantic, seg_map_semantic, len(self.CLASSES))
            edge_pre_eval_results = pre_eval_all_semantic_metric(
                pred_edge, seg_map_edge, 2)

            # instance metric calculation
            aji_metric = aggregated_jaccard_index(
                pred_instance, seg_map_instance, is_semantic=False)

            single_loop_results = dict(
                Aji=aji_metric,
                semantic_pre_eval_results=semantic_pre_eval_results,
                edge_pre_eval_results=edge_pre_eval_results)
            pre_eval_results.append(single_loop_results)

            data_id = self.data_infos[index]['ann_name'].replace(
                self.ann_suffix, '')
            # illustrating semantic level results
            if show_semantic:
                data_info = self.data_infos[index]
                image_path = osp.join(self.img_dir, data_info['img_name'])
                draw_semantic(
                    show_folder,
                    data_id,
                    image_path,
                    pred_semantic_edge,
                    seg_map_semantic_edge,
                    single_loop_results,
                    edge_id=self.EDGE_ID)

            # illustrating instance level results
            if show_instance:
                draw_instance(show_folder, data_id, pred_instance,
                              seg_map_instance)

        return pre_eval_results

    def model_agnostic_postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        id_list = list(np.unique(pred))
        # XXX: remove stuff prediction (we want to use stuff prediction to
        # enhance things prediction.)
        pred[pred > 79] = 0
        pred_canvas = np.zeros_like(pred).astype(np.uint8)
        for id in id_list:
            if id == self.ignore_index:
                continue
            id_mask = pred == id
            # fill instance holes
            id_mask = binary_fill_holes(id_mask)
            # remove small instance
            id_mask = remove_small_objects(id_mask, 20)
            id_mask = id_mask.astype(np.uint8)
            pred_canvas[id_mask > 0] = id
        pred_semantic = pred_canvas.copy()

        # TODO: Consider the dilation in postprocess (may be unnecessary in
        # natural instance segmentation)
        # pred_semantic = morphology.dilation(
        #     pred_semantic, selem=morphology.disk(2))

        # instance process & dilation
        pred_instance = measure.label(pred_canvas != self.ignore_index)
        # if re_edge=True, dilation pixel length should be 2
        pred_instance = morphology.dilation(
            pred_instance, selem=morphology.disk(2))

        return pred_semantic, pred_instance

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

        # convert pre_eval results to metric value
        semantic_pre_eval_results = ret_metrics.pop(
            'semantic_pre_eval_results')
        edge_pre_eval_results = ret_metrics.pop('edge_pre_eval_results')

        semantic_ret_metrics = pre_eval_to_metrics(semantic_pre_eval_results,
                                                   metric)
        edge_ret_metrics = pre_eval_to_metrics(edge_pre_eval_results, metric)
        for key, val in semantic_ret_metrics.items():
            ret_metrics[key] = val[1:]
        for key, val in edge_ret_metrics.items():
            ret_metrics['edge_' + key] = val[1]

        ret_metrics_per_class = {}
        ret_metrics_total = {}
        # calculate average metric
        for key in ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            # average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
            if 'Aji' in key:
                average_value = sum(ret_metrics[key]) / len(ret_metrics[key])
                ret_metrics_total[key] = np.round(average_value * 100, 2)
            elif key in metric:
                ret_metrics_total['m' + key] = np.round(
                    np.nanmean(ret_metrics[key]) * 100, 2)
                ret_metrics_per_class[key] = np.round(ret_metrics[key] * 100,
                                                      2)
            elif 'edge' in key:
                ret_metrics_total[key] = np.round(ret_metrics[key] * 100, 2)

        # convert to orderdict
        ret_metrics_per_class = OrderedDict(ret_metrics_per_class)
        ret_metrics_total = OrderedDict(ret_metrics_total)

        # per class metric show (remove background & edge)
        class_names = list(self.CLASSES)
        class_names.remove('background')
        class_names.remove('edge')
        ret_metrics_per_class.update({'classes': class_names})
        ret_metrics_per_class.move_to_end('classes', last=False)

        classes_table_data = PrettyTable()
        for key, val in ret_metrics_per_class.items():
            classes_table_data.add_column(key, val)

        print_log('Per class:', logger)
        print_log('\n' + classes_table_data.get_string(), logger=logger)

        # total metric show
        total_table_data = PrettyTable()
        for key, val in ret_metrics_total.items():
            total_table_data.add_column(key, [val])

        print_log('Total:', logger)
        print_log('\n' + total_table_data.get_string(), logger=logger)

        eval_results = {}
        # average results
        for key, value in ret_metrics_total.items():
            eval_results.update({key: value})

        ret_metrics_per_class.pop('classes', None)
        for key, value in ret_metrics_per_class.items():
            eval_results.update({
                key + '.' + str(name): f'{value[idx]:.3f}'
                for idx, name in enumerate(class_names)
            })

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
