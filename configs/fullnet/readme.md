# Full Resolution Network

[Improving nuclei/gland instance segmentation in histopathology images by full resolution neural network and spatial constrained loss](https://link.springer.com/content/pdf/10.1007/978-3-030-32239-7_42.pdf)

## Results

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--      | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 60.15 | 82.34 | 78.36 | 79.06 | 61.95 |

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | -    | 51.63  | 68.95  | 66.32 | 82.7  | 55.33 |
