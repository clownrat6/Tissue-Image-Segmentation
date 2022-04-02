# Multi-task U-Net

The multi-task U-Net with inner aware learning.

Multi-task:
1. semantic segmentation (background, semantic class 1, semantic class 2, ...);
2. inner aware segmentation (background, foreground - 1px);

## Results

### MoNuSeg (kumar)

| Method          | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--             | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| Multi-task UNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 82.58 | 55.0  | 73.91 | 78.91 | 58.33 | 82.36   | 56.29  | 71.51 | 77.69 | 55.68 | 

### CoNIC

| Method          | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--             | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| Multi-task UNet | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 68.96 | 51.65 | 65.74 | 80.77 | 53.47 |
