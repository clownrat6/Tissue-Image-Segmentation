from .bound_map import BoundLabelMake
from .direction_map import DirectionLabelMake
from .distance_map import DistanceLabelMake
from .hv_map import HVLabelMake
from .formating import (format_img, format_info, format_reg, format_seg, format_, to_tensor, Formatting)
from .transform import (AlbuColorJitter, ColorJitter, RandomFlip, Resize, RandomElasticDeform, RandomCrop, RandomRotate,
                        RandomSparseRotate, Identity, RandomBlur, Normalize, Pad, Affine)

__all__ = [
    'ColorJitter', 'RandomFlip', 'DirectionLabelMake', 'GenBound', 'Resize', 'to_tensor', 'format_img', 'format_seg',
    'format_reg', 'format_info', 'RandomElasticDeform', 'RandomCrop', 'Identity', 'RandomRotate', 'RandomSparseRotate',
    'RandomBlur', 'Normalize', 'format_', 'Pad', 'DistanceLabelMake', 'HVLabelMake', 'Affine', 'AlbuColorJitter'
]

class_dict = {
    'AlbuColorJitter': AlbuColorJitter,
    'ColorJitter': ColorJitter,
    'RandomFlip': RandomFlip,
    'Resize': Resize,
    'RandomElasticDeform': RandomElasticDeform,
    'RandomCrop': RandomCrop,
    'RandomRotate': RandomRotate,
    'RandomSparseRotate': RandomSparseRotate,
    'RandomBlur': RandomBlur,
    'Normalize': Normalize,
    'Pad': Pad,
    'Affine': Affine,
    'Identity': Identity,
    'BoundLabelMake': BoundLabelMake,
    'DirectionLabelMake': DirectionLabelMake,
    'Formatting': Formatting,
}
