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

w/ DDM:

| Method           | Dilation | Erosion    | Bound Width | imwAji | imwDice | mAji  | mDice | BoundDice | BoundPrecision | BoundRecall |
| :--              | :--      | :--        | :--:        | :-:    | :--:    | :--:  | :--:  | :--:      | :--:           | :--:        |
| Multi-task CDNet | 0        | 1          | 1           | 58.76  | 81.92   | 58.06 | 82.24 | 30.63     | 29.15          | 32.27       |
| Multi-task CDNet | 0        | 2          | 2           | 60.49  | 81.47   | 60.46 | 81.56 | 45.21     | 44.76          | 45.67       |
| Multi-task CDNet | 0        | 3          | 3           | 61.54  | 81.92   | 61.75 | 82.29 | 56.41     | 56.95          | 55.88       |
| Multi-task CDNet | 0        | 4          | 4           | 61.78  | 81.91   | 61.94 | 82.14 | 62.18     | 63.12          | 61.27       |
| Multi-task CDNet | 1        | 1          | 2           | 57.79  | 81.47   | 57.78 | 81.81 | 46.24     | 46.43          | 46.04       |
| Multi-task CDNet | 2        | 2          | 4           | 60.79  | 82.26   | 61.01 | 82.66 | 63.26     | 63.96          | 62.58       |
| Multi-task CDNet | 3        | 3          | 6           | 62.28  | 82.43   | 62.93 | 82.94 | 72.17     | 73.43          | 70.96       |
| Multi-task CDNet | 2        | 4          | 6           | 61.68  | 81.98   | 62.02 | 82.33 | 71.11     | 72.52          | 69.76       |
| Multi-task CDNet | 4        | 4          | 8           | 61.46  | 81.71   | 61.85 | 82.22 | 76.57     | 78.59          | 74.66       |

w/o DDM:

| Method           | Dilation | Erosion    | Bound Width | imwAji | imwDice | mAji  | mDice | BoundDice | BoundPrecision | BoundRecall |
| :--              | :--      | :--        | :--:        | :-:    | :--:    | :--:  | :--:  | :--:      | :--:           | :--:        |
| Multi-task CDNet | 0        | 1          | 1           | 57.55  | 82.12   | 57.16 | 82.44 | 29.79     | 30.11          | 29.47       |
| Multi-task CDNet | 0        | 2          | 2           | 58.71  | 81.2    | 58.06 | 81.31 | 43.98     | 45.06          | 42.95       |
| Multi-task CDNet | 0        | 3          | 3           | 61.19  | 82.07   | 61.17 | 82.33 | 55.5      | 57.52          | 53.62       |
| Multi-task CDNet | 0        | 4          | 4           | 61.67  | 81.91   | 61.79 | 82.14 | 61.61     | 63.95          | 59.44       |
| Multi-task CDNet | 1        | 1          | 2           | 55.44  | 81.63   | 54.5  | 81.91 | 45.68     | 46.59          | 44.81       |
| Multi-task CDNet | 2        | 2          | 4           | 59.36  | 81.49   | 59.36 | 81.75 | 62.25     | 64.17          | 60.44       |
| Multi-task CDNet | 3        | 3          | 6           | 62.05  | 82.43   | 62.71 | 82.94 | 72.13     | 73.62          | 70.68       |
| Multi-task CDNet | 2        | 4          | 6           | 61.67  | 81.98   | 62.04 | 82.33 | 71.04     | 72.72          | 69.43       |
| Multi-task CDNet | 4        | 4          | 8           | 61.37  | 81.7    | 61.71 | 82.11 | 76.57     | 78.29          | 74.83       |
