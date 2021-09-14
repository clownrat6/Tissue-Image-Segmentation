# (old) CDNet: Centripetal Direction Network for Nuclear Instance Segmentation

## Introduction

<!-- [ALGORITHM] -->

```latex
None
```

## Results and models

### MoNuSeg

| Method | Backbone | Crop Size | Lr schd | Aji | Dice | Aji (ms+flip) | Dice (ms+flip) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Nuclei CDNet | Deeplab-ResNet50 (r50-d8) | 320x320 | 20000 (Adam-lr5e-4) | 62.76 | 78.45 | 66.43 | 81.47 |
| Nuclei CDNet | Deeplab-ResNet50 (r50-d8) | 320x320 | 20000 (Adam-lr1e-3) | 63.23 | 79.04 | 66.39 | 81.49 |
| Nuclei CDNet | Deeplab-ResNet101 (r101-d8) | 320x320 | 20000 (Adam-lr5e-4) | 63.96 | 79.69 | 65.22 | 81.46 |
| Nuclei CDNet | Deeplab-ResNet101 (r101-d8) | 320x320 | 20000 (Adam-lr1e-3) | 64.09 | 79.62 | 65.87 | 81.22 |
| Nuclei CDNet | VGG16 (vgg16) | 320x320 | 20000 (Adam-lr5e-4) | 62.78 | 78.85 | 65.99 | 81.62 |

### CPM17

| Method | Backbone | Crop Size | Lr schd | Aji | Dice | Aji (ms+flip) | Dice (ms+flip) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Nuclei CDNet | Deeplab-ResNet101 (r101-d8) | 320x320 | 20000 (Adam-lr5e-4) | 67.88 | 85.21 | 69.86 | 86.63 |
| Nuclei CDNet | VGG16 (vgg16) | 320x320 | 20000 (Adam-lr5e-4) | 68.41 | 84.79 | 70.58 | 86.11 |

### CoNSep

| Method | Backbone | Crop Size | Lr schd | Aji | Dice | Aji (ms+flip) | Dice (ms+flip) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Nuclei CDNet | Deeplab-ResNet101 (r101-d8) | 320x320 | 20000 (Adam-lr5e-4) | 45.92 | 78.49 | 50.58 | 81.0 |
| Nuclei CDNet | VGG16 (vgg16) | 320x320 | 20000 (Adam-lr5e-4) | 48.33 | 78.41 | 52.61 | 81.05 |
