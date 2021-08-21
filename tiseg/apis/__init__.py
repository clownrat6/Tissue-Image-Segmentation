from .test import multi_gpu_test, single_gpu_test
from .train import set_random_seed, train_segmentor

__all__ = [
    'set_random_seed', 'train_segmentor', 'single_gpu_test', 'multi_gpu_test'
]
