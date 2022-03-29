# Contour Aware MicroNet

The MicroNet with contour aware learning.

## Results

### MoNuSeg (kumar)

| Method     | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--        | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| CMicroNet  | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 60.83 | 83.47 | 78.8  | 79.04 | 62.28 |

### CoNIC

| Method     | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--        | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| CMicroNet  | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 56.78    | 43.16  | 61.3   | 59.69 | 73.36 | 43.83 |
