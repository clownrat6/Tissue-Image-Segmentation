import numpy as np
from scipy.ndimage.morphology import distance_transform_cdt

from ..utils import assign_sem_class_to_insts


class DistanceLabelMake(object):
    """build direction label & point label for any dataset."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def dist_wo_norm(self, bin_image):
        res = np.zeros_like(bin_image)
        for j in range(1, bin_image.max() + 1):
            one_cell = np.zeros_like(bin_image)
            one_cell[bin_image == j] = 1
            one_cell = distance_transform_cdt(one_cell)
            res[bin_image == j] = one_cell[bin_image == j]
        res = res.astype('uint8')
        return res

    def __call__(self, sem_gt, inst_gt):
        results = {}
        inst_ids_pre_class = assign_sem_class_to_insts(inst_gt, sem_gt, self.num_classes)
        # reduce background dimension
        dist_canvas = np.zeros((self.num_classes - 1, *inst_gt.shape), dtype=np.float32)
        for sem_id, inst_ids in inst_ids_pre_class.items():
            if sem_id == 0:
                continue
            for inst_id in inst_ids:
                dist = self.dist_wo_norm((inst_gt == inst_id).astype(np.uint8))
                dist = dist / np.max(dist)
                dist_canvas[sem_id - 1, inst_gt == inst_id] = 0
                dist_canvas[sem_id - 1] += dist

        results['dist_gt'] = dist_canvas

        return results
