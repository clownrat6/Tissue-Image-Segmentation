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
from tiseg.datasets.utils.draw import Drawer

from tiseg.utils import (pre_eval_all_semantic_metric, pre_eval_to_sem_metrics, pre_eval_bin_aji, pre_eval_aji,
                         pre_eval_bin_pq, pre_eval_pq, pre_eval_to_bin_aji, pre_eval_to_aji, pre_eval_to_bin_pq,
                         pre_eval_to_pq, pre_eval_to_imw_pq, pre_eval_to_imw_aji)
from tiseg.models.utils import generate_direction_differential_map
from .builder import DATASETS
from .nuclei_dataset_mapper import NucleiDatasetMapper
from .utils import (re_instance, mudslide_watershed, align_foreground, get_tc_from_inst, get_dir_from_inst,
                    assign_sem_class_to_insts)


@DATASETS.register_module()
class NucleiCoNICDatasetWithDirection(Dataset):
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

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

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

        self.process_cfg = process_cfg
        self.num_angles = self.process_cfg.get('num_angles', 8)
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
            img_file_name = self.data_infos[index]['file_name']
            img_name = osp.splitext(osp.basename(img_file_name))[0]
            sem_file_name = self.data_infos[index]['sem_file_name']
            # semantic level label make
            sem_gt = mmcv.imread(sem_file_name, flag='unchanged', backend='pillow')
            # instance level label make
            inst_file_name = self.data_infos[index]['inst_file_name']
            inst_gt = np.load(inst_file_name)
            inst_gt = re_instance(inst_gt)

            # metric calculation & post process codes:
            sem_pred = pred['sem_pred'].copy()
            dir_pred = pred['dir_pred_test']
            dir_gt = pred['dir_gt'].copy()
            if 'inst_pred' not in pred:
                if 'dir_pred' in pred:
                    dir_pred = pred['dir_pred']
                    tc_sem_pred = pred['tc_sem_pred']
                    sem_pred, inst_pred = self.model_agnostic_postprocess_w_dir(dir_pred, tc_sem_pred, sem_pred)
                elif 'tc_sem_pred' in pred:
                    tc_sem_pred = pred['tc_sem_pred']
                    sem_pred, inst_pred = self.model_agnostic_postprocess_w_tc(tc_sem_pred, sem_pred)
                else:
                    # remove edge
                    sem_pred[sem_pred == len(self.CLASSES)] = 0
                    # model-agnostic post process operations
                    sem_pred, inst_pred = self.model_agnostic_postprocess(sem_pred)
            else:
                sem_pred = sem_pred
                inst_pred = pred['inst_pred']

            # semantic metric calculation (remove background class)
            sem_pre_eval_res = pre_eval_all_semantic_metric(sem_pred, sem_gt, len(self.CLASSES))
            dir_pre_eval_res = pre_eval_all_semantic_metric(dir_pred, dir_gt, self.num_angles + 1)

            # make contiguous ids
            inst_pred = measure.label(inst_pred.copy())
            inst_gt = measure.label(inst_gt.copy())

            pred_id_list_per_class = assign_sem_class_to_insts(inst_pred, sem_pred, len(self.CLASSES))
            gt_id_list_per_class = assign_sem_class_to_insts(inst_gt, sem_gt, len(self.CLASSES))

            # instance metric calculation
            bin_aji_pre_eval_res = pre_eval_bin_aji(inst_pred, inst_gt)
            aji_pre_eval_res = pre_eval_aji(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                            len(self.CLASSES))
            bin_pq_pre_eval_res = pre_eval_bin_pq(inst_pred, inst_gt)
            pq_pre_eval_res = pre_eval_pq(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class,
                                          len(self.CLASSES))

            single_loop_results = dict(
                bin_aji_pre_eval_res=bin_aji_pre_eval_res,
                aji_pre_eval_res=aji_pre_eval_res,
                bin_pq_pre_eval_res=bin_pq_pre_eval_res,
                pq_pre_eval_res=pq_pre_eval_res,
                dir_pre_eval_res=dir_pre_eval_res,
                sem_pre_eval_res=sem_pre_eval_res)
            pre_eval_results.append(single_loop_results)

            if show:
                if 'tc_pred' in pred:
                    tc_sem_pred_ = tc_sem_pred.copy()
                else:
                    tc_sem_pred_ = pred['sem_pred']
                    bound = tc_sem_pred_ == len(self.CLASSES)
                    tc_sem_pred_[tc_sem_pred_ > 0] = 1
                    tc_sem_pred_[bound > 0] = 2
                tc_sem_gt = get_tc_from_inst(inst_gt)
                pred_collect = {'sem_pred': sem_pred, 'inst_pred': inst_pred, 'tc_sem_pred': tc_sem_pred_}
                gt_collect = {'sem_gt': sem_gt, 'inst_gt': inst_gt, 'tc_sem_gt': tc_sem_gt}

                if 'dir_pred' in pred:
                    dir_pred_ = dir_pred.copy()
                    dir_gt = get_dir_from_inst(inst_gt, num_angle_types=8)

                    ddm_pred = generate_direction_differential_map(dir_pred_, direction_classes=(8 + 1))[0]
                    ddm_gt = generate_direction_differential_map(dir_gt, direction_classes=(8 + 1))[0]

                    pred_collect.update({'dir_pred': dir_pred_, 'ddm_pred': ddm_pred})
                    gt_collect.update({'dir_gt': dir_gt, 'ddm_gt': ddm_gt})

                self.drawer = Drawer(show_folder, sem_palette=self.PALETTE)
                metrics = {
                    'pixel_TP': sem_pre_eval_res[0],
                    'pixel_FP': sem_pre_eval_res[2],
                    'pixel_FN': sem_pre_eval_res[3],
                    'inst_TP': pq_pre_eval_res[0],
                    'inst_FP': pq_pre_eval_res[1],
                    'inst_FN': pq_pre_eval_res[2],
                }
                metrics = {}
                if 'dir_pred' in pred_collect and 'dir_gt' in gt_collect:
                    self.drawer.draw_direction(img_name, img_file_name, pred_collect, gt_collect, metrics)
                else:
                    self.drawer.draw(img_name, img_file_name, pred_collect, gt_collect, metrics)

        return pre_eval_results

    def model_agnostic_postprocess_w_dir(self, dir_pred, tc_sem_pred, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        # fill instance holes
        bin_sem_pred = tc_sem_pred.copy()
        bin_sem_pred[bin_sem_pred == 2] = 0
        bin_sem_pred = binary_fill_holes(bin_sem_pred)
        # remove small instance
        bin_sem_pred = remove_small_objects(bin_sem_pred, 20)
        bin_sem_pred = bin_sem_pred.astype(np.uint8)

        sem_id_list = list(np.unique(sem_pred))
        sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = sem_pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            # remove small instance
            sem_id_mask = remove_small_objects(sem_id_mask, 20)
            sem_id_mask_dila = morphology.dilation(sem_id_mask, selem=morphology.disk(2))
            sem_canvas[sem_id_mask_dila > 0] = sem_id
        sem_pred = sem_canvas

        bin_sem_pred, bound = mudslide_watershed(bin_sem_pred, dir_pred, sem_pred > 0)

        bin_sem_pred = remove_small_objects(bin_sem_pred, 20)
        inst_pred = measure.label(bin_sem_pred, connectivity=1)
        inst_pred = align_foreground(inst_pred, sem_pred > 0, 20)

        return sem_pred, inst_pred

    def model_agnostic_postprocess_w_tc(self, tc_sem_pred, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(sem_pred))
        sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
        # import matplotlib.pyplot as plt
        # plt.figure(dpi=300)
        # plt.subplot(121)
        # plt.imshow()
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = sem_pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_canvas[sem_id_mask > 0] = sem_id
        # plt.subplot(122)
        # plt.imshow(sem_canvas)
        # plt.savefig('2.png')
        # exit(0)

        # instance process & dilation
        bin_sem_pred = tc_sem_pred.copy()
        bin_sem_pred[bin_sem_pred == 2] = 0

        inst_pred = measure.label(bin_sem_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        # inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))
        inst_pred = align_foreground(inst_pred, sem_canvas > 0, 20)

        return sem_pred, inst_pred

    def model_agnostic_postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(pred))
        inst_pred = np.zeros_like(pred).astype(np.int32)
        sem_pred = np.zeros_like(pred).astype(np.uint8)
        cur = 0
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            inst_sem_mask = measure.label(sem_id_mask)
            inst_sem_mask = morphology.dilation(inst_sem_mask, selem=morphology.disk(2))
            inst_sem_mask[inst_sem_mask > 0] += cur
            inst_pred[inst_sem_mask > 0] = 0
            inst_pred += inst_sem_mask
            cur += len(np.unique(inst_sem_mask))
            sem_pred[inst_sem_mask > 0] = sem_id

        return sem_pred, inst_pred

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

        total_inst_keys = [
            'bAji', 'imwAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ'
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
            {'dir_m' + dir_key: np.round(np.mean(value) * 100, 2)
             for dir_key, value in dir_metrics.items()})

        dir_classes_metrics.update({'classes': list(range(1, self.num_angles + 1))})
        dir_classes_metrics.move_to_end('classes', last=False)

        dir_classes_table_data = PrettyTable()
        for key, val in dir_classes_metrics.items():
            dir_classes_table_data.add_column(key, val)

        total_dir_table_data = PrettyTable()
        for key, val in total_dir_metrics.items():
            total_dir_table_data.add_column(key, [val])

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
        print_log('\n' + total_dir_table_data.get_string(), logger=logger)

        storage_results = {
            'total_sem_metrics': total_sem_metrics,
            'total_inst_metrics': total_inst_metrics,
            'class_inst_metrics': classes_metrics,
            'total_dir_metrics': total_dir_metrics,
            'classes_dir_metrics': dir_classes_metrics
        }

        eval_results = {}
        # average results
        for k, v in total_sem_metrics.items():
            eval_results[k] = v
        for k, v in total_inst_metrics.items():
            eval_results[k] = v
        for k, v in total_dir_metrics.items():
            eval_results[k] = v

        classes = classes_metrics.pop('classes', None)
        for key, value in classes_metrics.items():
            eval_results.update({key + '.' + str(name): f'{value[idx]:.3f}' for idx, name in enumerate(classes)})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results, storage_results
