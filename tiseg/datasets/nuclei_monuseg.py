from .builder import DATASETS
from .nuclei_custom import NucleiCustomDataset


@DATASETS.register_module()
class NucleiMoNuSegDataset(NucleiCustomDataset):
    """MoNuSeg Nuclei Segmentation Dataset.

    MoNuSeg is actually instance segmentation task dataset. However, it can
    seem as a three class semantic segmentation task (Background, Nuclei, Edge)
    """

    CLASSES = ('background', 'nuclei', 'edge')

    PALETTE = [[0, 0, 0], [255, 2, 255], [2, 255, 255]]

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.tif', ann_suffix='_semantic_with_edge.png', **kwargs)
