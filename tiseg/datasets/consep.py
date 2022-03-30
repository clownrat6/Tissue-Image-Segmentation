from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CoNSePDataset(CustomDataset):
    """CoNSeP Nuclei segmentation dataset."""

    CLASSES = ('background', 'nuclei')

    PALETTE = [[0, 0, 0], [255, 2, 255]]

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', sem_suffix='_sem.png', inst_suffix='_inst.npy', **kwargs)
