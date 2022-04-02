# Full Resolution Network

[Improving nuclei/gland instance segmentation in histopathology images by full resolution neural network and spatial constrained loss](https://link.springer.com/content/pdf/10.1007/978-3-030-32239-7_42.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method  | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--     | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| FullNet | 256x256   | 8          | 40x40      | Adam-Lr1e-3   | 82.00 | 60.04 | 77.84 | 79.01 | 61.5  | 81.6    | 59.74  | 75.03 | 77.6  | 58.41 | 

### CoNIC

| Method  | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--     | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| FullNet | 256x256   | 8          | 40x40      | Adam-Lr1e-3   | 66.38 | 48.94 | 63.67 | 79.89 | 51.05 |
