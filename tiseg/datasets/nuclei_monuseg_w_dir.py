from .builder import DATASETS
from .nuclei_custom_w_dir import NucleiCustomDatasetWithDirection


@DATASETS.register_module()
class NucleiMoNuSegDatasetWithDirection(NucleiCustomDatasetWithDirection):
    """MoNuSeg Nuclei Segmentation Dataset.

    MoNuSeg is actually instance segmentation task dataset. However, it can
    seem as a two class semantic segmentation task (Background, Nuclei)
    """

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.tif', sem_suffix='_semantic.png', inst_suffix='_instance.npy', **kwargs)
