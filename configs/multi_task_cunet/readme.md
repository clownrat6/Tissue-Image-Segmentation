# Multi-task Contour Aware U-Net

The multi-task U-Net with contour aware learning.

Multi-task:
1. semantic segmentation (background, semantic class 1, semantic class 2, ...);
2. contour aware segmentation (background, foreground, contour);

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--              | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| Multi-task CUNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 81.83 | 60.79 | 76.66 | 78.97 | 60.54 | 81.29   | 60.31  | 74.11 | 77.66 | 57.74 | 

### CoNIC

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--              | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| Multi-task CUNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 66.9  | 49.2  | 62.67 | 80.38 | 50.75 |

## Empirical Study

> The metrics value is the average of the last five epochs.

### Boundary Width

| Method           | Dilation | Erosion    | Bound Width | imwAji | imwDice | mAji  | mDice | BoundDice | BoundPrecision | BoundRecall |
| :--              | :--      | :--        | :--:        | :-:    | :--:    | :--:  | :--:  | :--:      | :--:           | :--:        |
| Multi-task CUNet | 0        | 1          | 1           | 52.33  | 81.98   | 49.06 | 82.42 | 24.41     | 33.93          | 19.06       |
| Multi-task CUNet | 0        | 2          | 2           | 57.73  | 81.8    | 57.55 | 82.12 | 44.49     | 49.27          | 40.56       |
| Multi-task CUNet | 0        | 3          | 3           | 60.92  | 81.89   | 61.16 | 82.2  | 54.51     | 56.99          | 52.24       |
| Multi-task CUNet | 0        | 4          | 4           | 61.01  | 82.29   | 61.21 | 82.8  | 62.59     | 65.52          | 59.91       |
| Multi-task CUNet | 1        | 1          | 2           | 51.16  | 81.23   | 46.98 | 81.77 | 44.79     | 48.79          | 41.4        |
| Multi-task CUNet | 2        | 2          | 4           | 57.85  | 82.16   | 57.46 | 82.61 | 66.3      | 65.27          | 61.45       |
| Multi-task CUNet | 3        | 3          | 6           | 60.31  | 81.29   | 60.79 | 81.83 | 71.08     | 73.31          | 68.99       |
| Multi-task CUNet | 2        | 4          | 6           | 60.99  | 81.82   | 61.24 | 82.18 | 71.48     | 73.68          | 69.4        |
| Multi-task CUNet | 4        | 4          | 8           | 61.58  | 82.15   | 61.69 | 82.38 | 76.96     | 78.45          | 75.53       |


### Data Augmentation

| Method | Geometry Distortion | Color Distortion | Blur | imwAji | imwDice | mAji  | mDice |
| :--    | :--:                | :--:             | :--: | :-:    | :--:    | :--:  | :--:  |
| CUNet  |                     |                  |      | 60.38  | 82.89   | 60.14 | 83.58 |
| CUNet  | RandomFlip          |                  |      | 60.34  | 82.44   | 59.98 | 83.13 |
| CUNet  | Affine              |                  |      | 60.02  | 81.79   | 59.52 | 82.17 |
| CUNet  |                     | √                |      | 60.27  | 81.37   | 60.64 | 82.06 |
| CUNet  |                     |                  | √    | 60.48  | 82.86   | 59.85 | 83.48 |
| CUNet  |                     | √                | √    | 61.16  | 82.15   | 61.66 | 82.81 |
| CUNet  | √                   | √                | √    | 60.31  | 81.29   | 60.79 | 81.83 |
