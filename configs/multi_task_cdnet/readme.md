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
