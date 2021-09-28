from .builder import DATASETS
from .nuclei_custom import NucleiCustomDataset


@DATASETS.register_module()
class CoNSePDataset(NucleiCustomDataset):
    """CoNSeP segmentation dataset."""

    CLASSES = ('background', 'nuclei', 'edge')

    PALETTE = [[0, 0, 0], [255, 2, 255], [2, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png', ann_suffix='semantic_with_edge.png', **kwargs)
