from collections import OrderedDict

import mmcv
import numpy as np
import torch
from skimage import measure


def aggregated_jaccard_index(pred_label, target_label, is_semantic=True):
    """Aggregated Jaccard Index Calculation.

    It's only support binary mask now.

    Args:
        pred_label (numpy.ndarray): Prediction segmentation map.
        target_label (numpy.ndarray): Ground truth segmentation map.
        is_semantic (bool): If the input is semantic level. Default: True
    """
    pred_label = pred_label.copy()
    target_label = target_label.copy()

    if is_semantic:
        pred_label[pred_label > 0] = 1
        target_label[target_label > 0] = 1
        pred_label = measure.label(pred_label)
        target_label = measure.label(target_label)

    target_id_list = list(np.unique(target_label))
    pred_id_list = list(np.unique(pred_label))

    # Remove background class
    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = (pred_label == p).astype(np.uint8)
        pred_masks.append(p_mask)

    target_masks = [
        None,
    ]
    for t in target_id_list[1:]:
        t_mask = (target_label == t).astype(np.uint8)
        target_masks.append(t_mask)

    # prefill with value
    pairwise_intersection = np.zeros(
        [len(target_id_list) - 1,
         len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(target_id_list) - 1,
                               len(pred_id_list) - 1],
                              dtype=np.float64)

    # caching pairwise
    for target_id in target_id_list[1:]:  # 0-th is background
        t_mask = target_masks[target_id]
        pred_target_overlap = pred_label[t_mask > 0]
        pred_target_overlap_id = list(np.unique(pred_target_overlap))
        for pred_id in pred_target_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            intersect = (t_mask * p_mask).sum()
            pairwise_intersection[target_id - 1, pred_id - 1] = intersect
            pairwise_union[target_id - 1, pred_id - 1] = total - intersect

    pairwise_iou = pairwise_intersection / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each target, dont care
    # about reusing pred instance multiple times
    if pairwise_iou.shape[0] * pairwise_iou.shape[1] == 0:
        return 0.
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_target = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_target]
    overall_inter = (pairwise_intersection[paired_target, paired_pred]).sum()
    overall_union = (pairwise_union[paired_target, paired_pred]).sum()

    paired_target = list(paired_target + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # It seems that only unpaired Predictions need to be added into union.
    # unpaired_target = np.array(
    #     [idx for idx in target_id_list[1:] if idx not in paired_target])
    # for target_id in unpaired_target:
    #     overall_union += target_masks[target_id].sum()
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


def intersect_and_union(pred_label, label, num_classes):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results, gt_seg_maps, num_classes):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                results[i], gt_seg_maps[i], num_classes)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 metrics=['mIoU'],
                 nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = [
        'mIoU',
    ]
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics
