import numpy as np
from skimage import morphology


class BoundLabelMake:
    """Generate high quality boundary labels.

    The return is fixed to a three-class map (background, foreground, boundary).
    """

    def __init__(self, edge_id=2, selem_radius=3):
        self.edge_id = edge_id
        if isinstance(selem_radius, int):
            selem_radius = (selem_radius, selem_radius)
        self.radius = selem_radius

    def __call__(self, data):
        """generate boundary label from instance map and pure semantic map.

        sem_map:
            0: background
            1: semantic_class 1
            2: semantic class 2
            ...

        inst_map:
            0: background
            1: instance 1
            2: instance 2
            ...

        sem_map_w_bound:
            0: background
            1: foreground
            2: boundary

        Args:
            sem_map: two-class or multi-class semantic map without edge which is
                the raw semantic map.
            inst_map: instance map with each instance id. Use inst_map = inst_id
                to extrach each instance.
        """

        sem_gt = data['sem_gt']
        inst_gt = data['inst_gt']

        sem_gt_w_bound = np.zeros_like(sem_gt)
        sem_gt_w_bound += sem_gt

        # NOTE: sem_map must have same size as inst_map
        assert np.allclose(sem_gt > 0, inst_gt > 0)
        inst_id_list = list(np.unique(inst_gt))
        for inst_id in inst_id_list:
            if inst_id == 0:
                continue
            inst_id_mask = inst_gt == inst_id
            bound = morphology.dilation(
                inst_id_mask, selem=morphology.selem.diamond(self.radius[0])) & (
                    ~morphology.erosion(inst_id_mask, selem=morphology.selem.diamond(self.radius[1])))
            # bound = inst_id_mask & ~morphology.erosion(inst_id_mask, selem=morphology.selem.diamond(self.radius))
            sem_gt_w_bound[bound > 0] = self.edge_id

        # NOTE: sem_map is raw semantic map (two-class or multi-class without boundary)
        # NOTE: sem_map_w_bound has an extra semantic class (background, class1, class2, ..., bound)
        data['sem_gt_w_bound'] = sem_gt_w_bound
        data['seg_fields'].append('sem_gt_w_bound')

        return data
