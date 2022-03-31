# Multi-task Centripetal Direction Network

The multi-task CDNet:
1. direction prediction (background, angle 45°, angle 90°, ...);
2. contour aware segmentation (background, foreground, contour);
3. semantic segmentation (background, semantic class 1, semantic class 2, ...);

## Results

### MoNuSeg (kumar)

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--              | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| Multi-task CDNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 63.66 | 84.15 | 80.09 | 79.88 | 63.98 |

### CoNIC

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--              | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| Multi-task CUNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 68.13    | 52.7   | 69.86  | 66.47 | 81.66 | 54.58 |
