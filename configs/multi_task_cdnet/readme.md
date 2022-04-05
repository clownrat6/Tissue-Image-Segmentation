# Multi-task Centripetal Direction Network

The multi-task CDNet:
1. direction prediction (background, angle 45°, angle 90°, ...);
2. contour aware segmentation (background, foreground, contour);
3. semantic segmentation (background, semantic class 1, semantic class 2, ...);

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--              | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| Multi-task CDNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 82.47 | 62.22 | 78.22 | 79.55 | 62.23 | 82.08   | 61.81  | 75.35 | 78.10 | 59.03 | 

### CoNIC

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--              | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| Multi-task CDNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 69.57 | 52.43 | 66.18 | 81.53 | 54.28 |

### Boundary Width

> conference version multi_task cdnet

| Method           | Dilation | Erosion    | Bound Width | imwAji | imwDice | mAji  | mDice | BoundDice | BoundPrecision | BoundRecall |
| :--              | :--      | :--        | :--:        | :-:    | :--:    | :--:  | :--:  | :--:      | :--:           | :--:        |
| Multi-task CDNet | 0        | 1          | 1           | 58.76  | 81.92   | 58.06 | 82.24 | 30.63     | 29.15          | 32.27       |
| Multi-task CDNet | 0        | 2          | 2           | 60.49  | 81.47   | 60.46 | 81.56 | 45.21     | 44.76          | 45.67       |
| Multi-task CDNet | 0        | 3          | 3           | 61.54  | 81.92   | 61.75 | 82.29 | 56.41     | 56.95          | 55.88       |
| Multi-task CDNet | 0        | 4          | 4           | 61.78  | 81.91   | 61.94 | 82.14 | 62.18     | 63.12          | 61.27       |
| Multi-task CDNet | 1        | 1          | 2           | 57.79  | 81.47   | 57.78 | 81.81 | 46.24     | 46.43          | 46.04       |
| Multi-task CDNet | 2        | 2          | 4           | 60.79  | 82.26   | 61.01 | 82.66 | 63.26     | 63.96          | 62.58       |
| Multi-task CDNet | 3        | 3          | 6           | 62.28  | 82.43   | 62.93 | 82.94 | 72.17     | 73.43          | 70.96       |
| Multi-task CDNet | 2        | 4          | 6           | -      | -       | -     | 82.18 | 71.48     | 73.68          | 69.4        |
| Multi-task CDNet | 4        | 4          | 8           | 61.46  | 81.71   | 61.85 | 82.22 | 76.57     | 78.59          | 74.66       |
