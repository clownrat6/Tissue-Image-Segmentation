# Contour Aware MicroNet

The MicroNet with contour aware learning.

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method    | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--       | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| CMicroNet | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 82.32 | 57.20 | 78.27 | 79.08 | 61.90 | 81.99   | 59.09  | 75.34 | 77.72 | 58.73 | 

### CoNIC

| Method    | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--       | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| CMicroNet | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 60.71 | 42.73 | 59.09 | 73.20 | 43.32 |
