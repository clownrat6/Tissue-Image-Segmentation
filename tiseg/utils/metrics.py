from collections import OrderedDict

import mmcv
import numpy as np
import torch
from skimage import measure


# TODO: Add doc string & comments
def pre_eval_all_semantic_metric(pred_label, target_label, num_classes, ignore_index=255):
    """Generate pre eval results for all semantic metrics."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    mask = target_label != ignore_index
    pred_label = pred_label[mask]
    target_label = target_label[mask]

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    TN_per_class = Pred_per_class.sum() - (TP_per_class + FP_per_class + FN_per_class)

    ret_package = (TP_per_class, TN_per_class, FP_per_class, FN_per_class, Pred_per_class, GT_per_class)

    return ret_package


def pre_eval_aji(inst_pred, inst_gt):
    # make instance id contiguous
    inst_pred = measure.label(inst_pred.copy())
    inst_gt = measure.label(inst_gt.copy())

    pred_id_list = list(np.unique(inst_pred))
    gt_id_list = list(np.unique(inst_gt))

    if 0 not in pred_id_list:
        pred_id_list.insert(0, 0)

    if 0 not in gt_id_list:
        gt_id_list.insert(0, 0)

    # Remove background class
    pred_masks = []
    for p in pred_id_list:
        p_mask = (inst_pred == p).astype(np.uint8)
        pred_masks.append(p_mask)

    gt_masks = []
    for g in gt_id_list:
        g_mask = (inst_gt == g).astype(np.uint8)
        gt_masks.append(g_mask)

    # prefill with value
    pairwise_intersection = np.zeros([len(gt_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(gt_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for gt_id in gt_id_list:  # 0-th is background
        if gt_id == 0:
            continue
        g_mask = gt_masks[gt_id]
        pred_target_overlap = inst_pred[g_mask > 0]
        pred_target_overlap_id = list(np.unique(pred_target_overlap))
        for pred_id in pred_target_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (g_mask + p_mask).sum()
            intersect = (g_mask * p_mask).sum()
            pairwise_intersection[gt_id - 1, pred_id - 1] = intersect
            pairwise_union[gt_id - 1, pred_id - 1] = total - intersect

    pairwise_iou = pairwise_intersection / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each target, dont care
    # about reusing pred instance multiple times
    if pairwise_iou.shape[0] * pairwise_iou.shape[1] == 0:
        return 0., 0.
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_gt = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_gt]
    overall_inter = (pairwise_intersection[paired_gt, paired_pred]).sum()
    overall_union = (pairwise_union[paired_gt, paired_pred]).sum()

    paired_gt = list(paired_gt + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # It seems that only unpaired Predictions need to be added into union.
    unpaired_gt = np.array([idx for idx in gt_id_list if (idx not in paired_gt) and (idx != 0)])
    for gt_id in unpaired_gt:
        overall_union += gt_masks[gt_id].sum()
    unpaired_pred = np.array([idx for idx in pred_id_list if (idx not in paired_pred) and (idx != 0)])
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    return overall_inter, overall_union


def pre_eval_maji(inst_pred, inst_gt, sem_pred, sem_gt, num_classes):
    # make instance id contiguous
    inst_pred = measure.label(inst_pred.copy())
    inst_gt = measure.label(inst_gt.copy())

    pred_id_list = list(np.unique(inst_pred))
    gt_id_list = list(np.unique(inst_gt))

    if 0 not in pred_id_list:
        pred_id_list.insert(0, 0)

    if 0 not in gt_id_list:
        gt_id_list.insert(0, 0)

    def to_one_hot(mask, num_classes):
        ret = np.zeros((num_classes, *mask.shape))
        for i in range(num_classes):
            ret[i, mask == i] = 1

        return ret

    sem_pred_one_hot = to_one_hot(sem_pred, num_classes)
    sem_gt_one_hot = to_one_hot(sem_gt, num_classes)

    # Remove background class
    pred_masks = []
    pred_id_list_per_class = {}
    for p in pred_id_list:
        p_mask = (inst_pred == p).astype(np.uint8)

        tp = np.sum(p_mask * sem_pred_one_hot, axis=(-2, -1))
        belong_sem_id = np.argmax(tp)

        if belong_sem_id not in pred_id_list_per_class:
            pred_id_list_per_class[belong_sem_id] = [p]
        else:
            pred_id_list_per_class[belong_sem_id].append(p)

        pred_masks.append(p_mask)

    gt_masks = []
    gt_id_list_per_class = {}
    for g in gt_id_list:
        g_mask = (inst_gt == g).astype(np.uint8)

        tp = np.sum(g_mask * sem_gt_one_hot, axis=(-2, -1))
        belong_sem_id = np.argmax(tp)

        if belong_sem_id not in gt_id_list_per_class:
            gt_id_list_per_class[belong_sem_id] = [g]
        else:
            gt_id_list_per_class[belong_sem_id].append(g)

        gt_masks.append(g_mask)

    overall_inter = 0
    overall_union = 0
    for sem_id in gt_id_list_per_class.keys():
        if sem_id == 0 or sem_id not in pred_id_list_per_class:
            continue
        pred_inst_map = sum([((inst_pred == pred_id).astype(np.int32) * (idx + 1))
                             for idx, pred_id in enumerate(pred_id_list_per_class[sem_id])])
        gt_inst_map = sum([((inst_gt == gt_id).astype(np.int32) * (idx + 1))
                           for idx, gt_id in enumerate(gt_id_list_per_class[sem_id])])

        res = pre_eval_aji(pred_inst_map, gt_inst_map)
        overall_inter += res[0]
        overall_union += res[1]

    # calculate unpaired gt semantic class
    for sem_id in gt_id_list_per_class.keys():
        if sem_id in pred_id_list_per_class and sem_id != 0:
            continue
        gt_id_list = gt_id_list_per_class[sem_id]
        for gt_id in gt_id_list:
            if gt_id == 0:
                continue
            overall_union += np.sum(inst_gt == gt_id)

    # calculate unpaired pred semantic class
    for sem_id in pred_id_list_per_class.keys():
        if sem_id in gt_id_list_per_class and sem_id != 0:
            continue
        pred_id_list = pred_id_list_per_class[sem_id]
        for pred_id in pred_id_list:
            if pred_id == 0:
                continue
            overall_union += np.sum(inst_pred == pred_id)

    return overall_inter, overall_union


# NOTE: Another implementation of maji pre eval
# def pre_eval_maji(inst_pred, inst_gt, sem_pred, sem_gt, num_classes):
#     # make instance id contiguous
#     inst_pred = measure.label(inst_pred.copy())
#     inst_gt = measure.label(inst_gt.copy())

#     raw_pred_id_list = list(np.unique(inst_pred))
#     raw_gt_id_list = list(np.unique(inst_gt))

#     if 0 not in raw_pred_id_list:
#         raw_pred_id_list.insert(0, 0)

#     if 0 not in raw_gt_id_list:
#         raw_gt_id_list.insert(0, 0)

#     def to_one_hot(mask, num_classes):
#         ret = np.zeros((num_classes, *mask.shape))
#         for i in range(num_classes):
#             ret[i, mask == i] = 1

#         return ret

#     sem_pred_one_hot = to_one_hot(sem_pred, num_classes)
#     sem_gt_one_hot = to_one_hot(sem_gt, num_classes)

#     # Remove background class
#     pred_masks = []
#     pred_id_list_per_class = {}
#     for p in raw_pred_id_list:
#         p_mask = (inst_pred == p).astype(np.uint8)

#         tp = np.sum(p_mask * sem_pred_one_hot, axis=(-2, -1))
#         belong_sem_id = np.argmax(tp)

#         if belong_sem_id not in pred_id_list_per_class:
#             pred_id_list_per_class[belong_sem_id] = [p]
#         else:
#             pred_id_list_per_class[belong_sem_id].append(p)

#         pred_masks.append(p_mask)

#     gt_masks = []
#     gt_id_list_per_class = {}
#     for g in raw_gt_id_list:
#         g_mask = (inst_gt == g).astype(np.uint8)

#         tp = np.sum(g_mask * sem_gt_one_hot, axis=(-2, -1))
#         belong_sem_id = np.argmax(tp)

#         if belong_sem_id not in gt_id_list_per_class:
#             gt_id_list_per_class[belong_sem_id] = [g]
#         else:
#             gt_id_list_per_class[belong_sem_id].append(g)

#         gt_masks.append(g_mask)

#     # prefill with value
#     pairwise_intersection = np.zeros([len(raw_gt_id_list) - 1, len(raw_pred_id_list) - 1], dtype=np.float64)
#     pairwise_union = np.zeros([len(raw_gt_id_list) - 1, len(raw_pred_id_list) - 1], dtype=np.float64)
#     # caching pairwise
#     for sem_id in gt_id_list_per_class.keys():
#         if sem_id not in pred_id_list_per_class or sem_id == 0:
#             continue
#         gt_id_list = gt_id_list_per_class[sem_id]
#         pred_id_list = pred_id_list_per_class[sem_id]
#         for gt_id in gt_id_list:  # 0-th is background
#             if gt_id == 0:
#                 continue
#             g_mask = gt_masks[gt_id]
#             pred_target_overlap = inst_pred[g_mask > 0]
#             pred_target_overlap_id = list(np.unique(pred_target_overlap))
#             for pred_id in pred_target_overlap_id:
#                 if pred_id == 0:  # ignore
#                     continue  # overlaping background
#                 if pred_id not in pred_id_list:
#                     continue
#                 p_mask = pred_masks[pred_id]
#                 total = (g_mask + p_mask).sum()
#                 intersect = (g_mask * p_mask).sum()
#                 pairwise_intersection[gt_id - 1, pred_id - 1] = intersect
#                 pairwise_union[gt_id - 1, pred_id - 1] = total - intersect

#     pairwise_iou = pairwise_intersection / (pairwise_union + 1.0e-6)
#     # pair of pred that give highest iou for each target, dont care
#     # about reusing pred instance multiple times
#     if pairwise_iou.shape[0] * pairwise_iou.shape[1] == 0:
#         return 0., 0.
#     paired_pred = np.argmax(pairwise_iou, axis=1)
#     pairwise_iou = np.max(pairwise_iou, axis=1)
#     # exlude those dont have intersection
#     paired_gt = np.nonzero(pairwise_iou > 0.0)[0]
#     paired_pred = paired_pred[paired_gt]
#     overall_inter = (pairwise_intersection[paired_gt, paired_pred]).sum()
#     overall_union = (pairwise_union[paired_gt, paired_pred]).sum()

#     paired_gt = list(paired_gt + 1)  # index to instance ID
#     paired_pred = list(paired_pred + 1)
#     # It seems that only unpaired Predictions need to be added into union.
#     unpaired_gt = np.array([idx for idx in raw_gt_id_list if (idx not in paired_gt) and (idx != 0)])
#     for gt_id in unpaired_gt:
#         overall_union += gt_masks[gt_id].sum()

#     unpaired_pred = np.array([idx for idx in raw_pred_id_list if (idx not in paired_pred) and (idx != 0)])
#     for pred_id in unpaired_pred:
#         overall_union += pred_masks[pred_id].sum()

#     return overall_inter, overall_union


def accuracy(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class accuracy calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    TN_per_class = pred_label.numel() - (TP_per_class + FP_per_class + FN_per_class)

    accuracy = (TP_per_class + TN_per_class) / pred_label.numel()

    accuracy = np.nan_to_num(accuracy.numpy(), nan_to_num)

    return accuracy


def precision_recall(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class precision-recall calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    precision = TP_per_class / (TP_per_class + FP_per_class)
    recall = TP_per_class / (TP_per_class + FN_per_class)

    precision = np.nan_to_num(precision.numpy(), nan_to_num)
    recall = np.nan_to_num(recall.numpy(), nan_to_num)

    return precision, recall


def dice_similarity_coefficient(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class dice calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    dice = 2 * TP_per_class / (Pred_per_class + GT_per_class)

    dice = np.nan_to_num(dice.numpy(), nan_to_num)

    return dice


def intersect_and_union(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class iou calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    iou = TP_per_class / (Pred_per_class + GT_per_class - TP_per_class)

    iou = np.nan_to_num(iou.numpy(), nan_to_num)

    return iou


def aggregated_jaccard_index(inst_pred, inst_gt):
    """Aggregated Jaccard Index Calculation.

    0 is set as background pixels and we will ignored.

    Args:
        inst_pred (numpy.ndarray): Prediction instance map.
        inst_gt (numpy.ndarray): Ground truth instance map.
    """
    i, u = pre_eval_aji(inst_pred, inst_gt)
    if i == 0. or u == 0.:
        return 0.
    aji_score = i / u
    return aji_score


def mean_aggregated_jaccard_index(inst_pred, inst_gt, sem_pred, sem_gt, num_classes):
    """Class Wise Aggregated Jaccard Index Calculation.

    0 is set as background pixels and we will ignored.

    Args:
        inst_pred (numpy.ndarray): Prediction instance map.
        inst_gt (numpy.ndarray): Ground truth instance map.
    """
    i, u = pre_eval_maji(inst_pred, inst_gt, sem_pred, sem_gt, num_classes)
    if i == 0. or u == 0.:
        return 0
    aji_score = i / u
    return aji_score


def pre_eval_to_aji(pre_eval_results, nan_to_num=None):
    """Convert aji pre-eval overall intersection & pre-eval overall union to aji score."""
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 2

    # [0]: overall intersection
    # [1]: overall union
    ret_metrics = {'Aji': sum(pre_eval_results[0]) / sum(pre_eval_results[1])}

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics


def pre_eval_to_sem_metrics(pre_eval_results, metrics=['IoU'], nan_to_num=None, beta=1):
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

    ret_metrics = total_area_to_sem_metrics(total_area_TP, total_area_TN, total_area_FP, total_area_FN,
                                            total_area_pred_label, total_area_label, metrics, nan_to_num)

    return ret_metrics


def total_area_to_sem_metrics(total_area_TP,
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
            iou = total_area_TP / (total_area_pred_label + total_area_label - total_area_TP)
            ret_metrics['IoU'] = iou
        elif metric == 'Dice':
            dice = 2 * total_area_TP / (total_area_pred_label + total_area_label)
            ret_metrics['Dice'] = dice
        elif metric == 'Recall':
            recall = total_area_TP / (total_area_TP + total_area_FN)
            ret_metrics['Recall'] = recall
        elif metric == 'Precision':
            precision = total_area_TP / (total_area_TP + total_area_FP)
            ret_metrics['Precision'] = precision

    # convert torch to numpy
    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})
    return ret_metrics
