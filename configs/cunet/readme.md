# Contour Aware U-Net

The U-Net with contour aware learning (popular).

## Results

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| CUNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 62.38 | 83.31 | 79.4  | 78.95 | 62.69 |

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| CUNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 56.05    | 44.69  | 63.22  | 61.53 | 72.54 | 44.67 |

## Experiments

### Boundary Width

| Method | Dilation | Erosion    | Bound Width | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--      | :--        | :--:        | :-:   | :--:  | :--:  | :--:  | :--:  |
| CUNet  | 0        | 1          | 1           | 57.44 | 80.07 | 74.81 | 73.46 | 54.96 |
| CUNet  | 0        | 2          | 2           | 59.11 | 83.11 | 77.1  | 78.13 | 60.24 |
| CUNet  | 0        | 3          | 3           | 61.64 | 82.78 | 79.79 | 78.87 | 62.93 |
| CUNet  | 1        | 1          | 2           | 52.28 | 82.56 | 72.56 | 79.08 | 57.38 |
| CUNet  | 2        | 2          | 4           | 60.47 | 83.35 | 78.48 | 78.66 | 61.74 |
| CUNet  | 3        | 3          | 6           | 62.38 | 83.31 | 79.4  | 78.95 | 62.69 |

### Data Augmentation

| Method | Geometry Distortion | Color Distortion | Blur | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--:                | :--:             | :--: | :-:   | :--:  | :--:  | :--:  | :--:  |
| CUNet  |                     |                  |      | 60.74 | 82.34 | 78.36 | 79.1  | 61.98 |
| CUNet  | RandomFlip          |                  |      | 61.83 | 82.9  | 79.04 | 79.26 | 62.65 |
| CUNet  | RandomCrop          |                  |      | 60.74 | 82.48 | 77.87 | 79.06 | 61.56 |
| CUNet  | Affine              |                  |      | 60.82 | 83.05 | 78.23 | 79.56 | 62.23 |
| CUNet  |                     | √                |      | 59.91 | 82.22 | 77.37 | 79.57 | 61.56 |
| CUNet  |                     |                  | √    | 59.39 | 81.42 | 74.21 | 77.33 | 57.39 |
| CUNet  |                     | √                | √    | - | 81.42 | 74.21 | 77.33 | 57.39 |
| CUNet  | √                   | √                | √    | - | 81.42 | 74.21 | 77.33 | 57.39 |

