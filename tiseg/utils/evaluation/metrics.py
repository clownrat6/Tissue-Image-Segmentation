from collections import OrderedDict

import mmcv
import numpy as np
import torch
from skimage import measure


# TODO: Add doc string & comments
def pre_eval_all_semantic_metric(pred_label, target_label, num_classes):
    """Generate pre eval results for all semantic metrics."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(
            mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(
        TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(
        FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(
        FN.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(
        target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    TN_per_class = Pred_per_class.sum() - (
        TP_per_class + FP_per_class + FN_per_class)

    ret_package = (TP_per_class, TN_per_class, FP_per_class, FN_per_class,
                   Pred_per_class, GT_per_class)

    return ret_package


# TODO: Add doc string & comments
def accuracy(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class accuracy calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(
            mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(
        TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(
        FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(
        FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    TN_per_class = pred_label.numel() - (
        TP_per_class + FP_per_class + FN_per_class)

    accuracy = (TP_per_class + TN_per_class) / pred_label.numel()

    accuracy = np.nan_to_num(accuracy.numpy(), nan_to_num)

    return accuracy


# TODO: Add doc string & comments
def precision_recall(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class precision-recall calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(
            mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(
        TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(
        FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(
        FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    precision = TP_per_class / (TP_per_class + FP_per_class)
    recall = TP_per_class / (TP_per_class + FN_per_class)

    precision = np.nan_to_num(precision.numpy(), nan_to_num)
    recall = np.nan_to_num(recall.numpy(), nan_to_num)

    return precision, recall


# TODO: Add comments & doc string
def dice_similarity_coefficient(pred_label,
                                target_label,
                                num_classes,
                                nan_to_num=None):
    """multi-class dice calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(
            mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(
        TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(
        target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    dice = 2 * TP_per_class / (Pred_per_class + GT_per_class)

    dice = np.nan_to_num(dice.numpy(), nan_to_num)

    return dice


# TODO: Add comments & doc string
def intersect_and_union(pred_label,
                        target_label,
                        num_classes,
                        nan_to_num=None):
    """multi-class iou calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(
            mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(
        TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(
        target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    iou = TP_per_class / (Pred_per_class + GT_per_class - TP_per_class)

    iou = np.nan_to_num(iou.numpy(), nan_to_num)

    return iou


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
        pred_label[pred_label != 1] = 0
        target_label[target_label != 1] = 0
        pred_label = measure.label(pred_label)
        target_label = measure.label(target_label)

    pred_id_list = list(np.unique(pred_label))
    target_id_list = list(np.unique(target_label))

    # move zero element to the first place
    if 0 in pred_id_list:
        pred_id_list.remove(0)
        pred_id_list.insert(0, 0)
    if 0 in target_id_list:
        target_id_list.remove(0)
        target_id_list.insert(0, 0)

    # Remove background class
    pred_masks = {
        0: None,
    }
    for p in pred_id_list[1:]:
        p_mask = (pred_label == p).astype(np.uint8)
        pred_masks[p] = p_mask

    target_masks = {
        0: None,
    }
    for t in target_id_list[1:]:
        t_mask = (target_label == t).astype(np.uint8)
        target_masks[t] = t_mask

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
    unpaired_target = np.array(
        [idx for idx in target_id_list[1:] if idx not in paired_target])
    for target_id in unpaired_target:
        overall_union += target_masks[target_id].sum()
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['IoU'],
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
    assert len(pre_eval_results) == 6

    total_area_TP = sum(pre_eval_results[0])
    total_area_TN = sum(pre_eval_results[1])
    total_area_FP = sum(pre_eval_results[2])
    total_area_FN = sum(pre_eval_results[3])
    total_area_pred_label = sum(pre_eval_results[4])
    total_area_label = sum(pre_eval_results[5])

    ret_metrics = total_area_to_metrics(total_area_TP, total_area_TN,
                                        total_area_FP, total_area_FN,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num)

    return ret_metrics


def total_area_to_metrics(total_area_TP,
                          total_area_TN,
                          total_area_FP,
                          total_area_FN,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['IoU'],
                          nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        total_area_TP (torch.Tensor): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_FP (torch.Tensor): The false positive pixels histogram on
            all classes.
        total_area_FN (torch.Tensor): The prediction histogram on all
            classes.
        total_area_pred_label (torch.Tensor): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'IoU' and 'Dice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        dict: Contains selected metric value.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['Accuracy', 'IoU', 'Dice', 'Recall', 'Precision']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = {}
    for metric in metrics:
        if metric == 'Accuracy':
            acc = (total_area_TP + total_area_TN) / total_area_label.sum()
            ret_metrics['Accuracy'] = acc
        elif metric == 'IoU':
            iou = total_area_TP / (
                total_area_pred_label + total_area_label - total_area_TP)
            ret_metrics['IoU'] = iou
        elif metric == 'Dice':
            dice = 2 * total_area_TP / (
                total_area_pred_label + total_area_label)
            ret_metrics['Dice'] = dice
        elif metric == 'Recall':
            recall = total_area_TP / (total_area_TP + total_area_FN)
            ret_metrics['Recall'] = recall
        elif metric == 'Precision':
            precision = total_area_TP / (total_area_TP + total_area_FP)
            ret_metrics['Precision'] = precision

    # convert torch to numpy
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
