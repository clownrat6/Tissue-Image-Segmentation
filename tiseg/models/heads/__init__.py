from .cd_head import CDHead
from .unet_head import UNetHead
from .multi_task_unet_head import MultiTaskUNetHead
from .cd_voronoi_head import CDVoronoiHead
from .regression_cd_head import RegCDHead
from .regression_degree_cd_head import RegDegreeCDHead
from .multi_task_cd_head_no_point import MultiTaskCDHeadNoPoint
from .multi_task_cd_voronoi_head import MultiTaskCDVoronoiHead
from .multi_task_cd_head_twobranch import MultiTaskCDHeadTwobranch
__all__ = [
    'CDHead', 'UNetHead', 'CDVoronoiHead', 'MultiTaskUNetHead', 'RegCDHead', 'RegDegreeCDHead',
    'MultiTaskCDHeadNoPoint', 'MultiTaskCDVoronoiHead', 'MultiTaskCDHeadTwobranch'
]
