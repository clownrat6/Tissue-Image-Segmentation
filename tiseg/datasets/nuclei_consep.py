from .builder import DATASETS
from .nuclei_custom import NucleiCustomDataset


@DATASETS.register_module()
class NucleiCoNSePDataset(NucleiCustomDataset):
    """CoNSeP Nuclei segmentation dataset."""

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', ann_suffix='semantic.png', **kwargs)
