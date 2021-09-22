import numpy as np
from skimage import measure, morphology

label = np.zeros((224, 224))

raw_semantic_map = label
raw_semantic_map_inside = (raw_semantic_map == 1).astype(np.uint8)

semantic_map_with_edge = np.zeros((label.shape[0], label.shape[1]),
                                  dtype=np.uint8)
semantic_map_with_edge[raw_semantic_map_inside == 1] = 1

boun = morphology.dilation(
    semantic_map_with_edge, selem=morphology.selem.disk(1)) & (
        ~morphology.erosion(semantic_map_with_edge, morphology.disk(1)))
semantic_map_with_edge[boun > 0] = 2  # boundary

semantic_map_inside = (semantic_map_with_edge == 1).astype(np.uint8)
instance_map = measure.label(semantic_map_inside, connectivity=1)
instance_map = morphology.dilation(
    instance_map, selem=morphology.selem.disk(2))
semantic_map = (instance_map > 0).astype(np.uint8)
