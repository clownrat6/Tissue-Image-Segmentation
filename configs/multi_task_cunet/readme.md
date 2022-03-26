# Multi-task Contour Aware U-Net

The multi-task U-Net with contour aware learning.

Multi-task:
1. semantic segmentation (background, semantic class 1, semantic class 2, ...);
2. contour aware segmentation (background, foreground, contour);

## Results

### MoNuSeg (kumar)

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--              | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| Multi-task CUNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 61.95 | 83.21 | 77.81 | 79.25 | 61.67 |

### CoNIC

| Method           | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--              | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| Multi-task CUNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 63.06    | 49.82  | 67.54  | 63.48 | 80.47 | 51.44 |
