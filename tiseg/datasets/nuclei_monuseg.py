from .builder import DATASETS
from .nuclei_custom import NucleiCustomDataset


@DATASETS.register_module()
class NucleiMoNuSegDataset(NucleiCustomDataset):
    """MoNuSeg Nuclei Segmentation Dataset.

    MoNuSeg is actually instance segmentation task dataset. However, it can be
    seen as a two class semantic segmentation task (Background, Nuclei).
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.tif', sem_suffix='_sem.png', inst_suffix='_inst.npy', **kwargs)
