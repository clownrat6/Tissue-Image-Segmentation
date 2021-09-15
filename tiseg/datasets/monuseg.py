import os.path as osp
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

from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
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
                 draw_semantic=False,
                 draw_instance=False):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            draw_semantic (bool): Illustrate semantic level prediction &
                ground truth. Default: False
            draw_instance (bool): Illustrate instance level prediction &
                ground truth. Default: False

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        # make it accessable in a single evaluation loop for semantic results
        # drawing
        self.pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir,
                               self.data_infos[index]['ann_name'])
            seg_map = mmcv.imread(seg_map, flag='unchanged', backend='pillow')

            # metric calculation post process codes:

            # extract inside
            pred = (pred == 1).astype(np.uint8)
            seg_map = (seg_map == 1).astype(np.uint8)

            # model-agnostic post process operations
            pred_semantic, pred_instance = self.model_agnostic_postprocess(
                pred)
            # semantic metric calculation
            seg_map_semantic = seg_map.copy()
            TP = (pred_semantic == 1) * (seg_map_semantic == 1)
            FP = (pred_semantic == 1) * (seg_map_semantic == 0)
            FN = (pred_semantic == 0) * (seg_map_semantic == 1)
            Pred = (pred_semantic == 1)
            GT = (seg_map_semantic == 1)
            precision_metric = np.sum(TP) / (np.sum(TP) + np.sum(FP))
            recall_metric = np.sum(TP) / (np.sum(TP) + np.sum(FN))
            dice_metric = 2 * np.sum(TP) / (np.sum(Pred) + np.sum(GT))

            # instance metric calculation
            seg_map_instance = measure.label(seg_map)
            aji_metric = aggregated_jaccard_index(
                pred_instance, seg_map_instance, is_semantic=False)

            # TODO: (Important issue about post process)
            # This may be the dice metric calculation trick (Need be
            # considering carefully)
            # convert instance map (after postprocess) to semantic level
            # pred = (pred > 0).astype(np.uint8)
            # seg_map = (seg_map > 0).astype(np.uint8)

            self.pre_eval_results.append(
                dict(
                    aji=aji_metric,
                    dice=dice_metric,
                    recall=recall_metric,
                    precision=precision_metric))

            # illustrating semantic level results
            if draw_semantic:
                self.draw_semantic(pred_semantic, seg_map_semantic, index)

            # illustrating instance level results
            if draw_instance:
                self.draw_instance(pred_instance, seg_map_instance, index)

        return self.pre_eval_results

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
        pred_instance = morphology.dilation(
            pred_instance, selem=morphology.disk(1))

        return pred_semantic, pred_instance

    def draw_semantic(self, pred, label, index):
        """draw semantic level picture with FP & FN."""
        import matplotlib.pyplot as plt

        # Only support single sample inference now
        assert isinstance(index, int)

        plt.figure(figsize=(7 * 2, 7 * 2 + 4))

        # prediction drawing
        plt.subplot(221)
        plt.imshow(pred)
        plt.axis('off')
        plt.title('Prediction', fontsize=20, color='black')

        # ground truth drawing
        plt.subplot(222)
        plt.imshow(label)
        plt.axis('off')
        plt.title('Ground Truth', fontsize=20, color='black')

        # image drawing
        data_info = self.data_infos[index]
        data_id = osp.splitext(data_info['img_name'])[0]
        image_path = osp.join(self.img_dir, data_info['img_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(223)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Image', fontsize=20, color='black')

        canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
        canvas[label == 1, :] = (255, 255, 2)
        canvas[(pred == 0) * (label == 1), :] = (2, 255, 255)
        canvas[(pred == 1) * (label == 0), :] = (255, 2, 255)
        plt.subplot(224)
        plt.imshow(canvas)
        plt.axis('off')
        plt.title('FP-FN-Ground Truth', fontsize=20, color='black')

        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [(255, 255, 2), (2, 255, 255), (255, 2, 255)]
        label_list = [
            'Ground Truth',
            'TN',
            'FP',
        ]
        for color, label in zip(colors, label_list):
            color = list(color)
            color = [x / 255 for x in color]
            plt.plot(0, 0, '-', color=tuple(color), label=label)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3)

        # results visulization
        single_loop_results = self.pre_eval_results[0]
        aji = single_loop_results['aji']
        dice = single_loop_results['dice']
        recall = single_loop_results['recall']
        precision = single_loop_results['precision']
        print(f'aji: {aji}\ndice: '
              f'{dice}\nrecall: {recall}\nprecision: '
              f'{precision}')
        temp_str = (f'aji: {aji:.2f}\ndice: '
                    f'{dice:.2f}\nrecall: {recall:.2f}\nprecision: '
                    f'{precision:.2f}')
        plt.suptitle(temp_str, fontsize=20, color='black')
        plt.tight_layout()
        plt.savefig(f'{data_id}_monuseg_semantic_compare.png', dpi=400)

    def draw_instance(self, pred_instance, label_instance, index):
        """draw instance level picture."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7 * 2, 7))

        data_info = self.data_infos[index]
        data_id = osp.splitext(data_info['img_name'])[0]

        plt.subplot(121)
        plt.imshow(pred_instance)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(label_instance)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{data_id}_monuseg_instance_compare.png', dpi=400)

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
        allowed_metrics = ['aji', 'dice', 'all']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        assert ('all' in metric) and (dump_path is not None)

        # clear pre-eval results
        self.pre_eval_results.clear()

        eval_results = {}
        ret_metrics = {}

        # test a list of files
        if 'all' in metric:
            aji_list = []
            dice_list = []
            for item in results:
                aji_list.append(item['aji'])
                dice_list.append(item['dice'])
            ret_metrics['aji'] = np.array([sum(aji_list) / len(aji_list)])
            ret_metrics['dice'] = np.array([sum(dice_list) / len(dice_list)])

            # TODO: Refactor for more general metric
            if dump_path is not None:
                name_list = self.data_infos
                fp = open(f'{dump_path}', 'w')
                fp.write(f'{"filename":<30} | {"aji":<30} | {"dice":<30}')
                for data_info, aji, dice in zip(name_list, aji_list,
                                                dice_list):
                    name = data_info['ann_name'].split('_')[0]
                    aji = f'{aji * 100:.2f}'
                    dice = f'{dice * 100:.2f}'
                    fp.write(f'{name:<30} | {aji:<30} | {dice:<30}')

        if 'aji' in metric:
            aji_list = []
            for item in results:
                aji_list.append(item['aji'])
            ret_metrics['aji'] = np.array([sum(aji_list) / len(aji_list)])
        if 'dice' in metric:
            dice_list = []
            for item in results:
                dice_list.append(item['dice'])
            ret_metrics['dice'] = np.array([sum(dice_list) / len(dice_list)])

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

        if 'aji' in ret_metrics:
            eval_results['aji'] = ret_metrics['aji'][0]
        if 'dice' in ret_metrics:
            eval_results['dice'] = ret_metrics['dice'][0]

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
