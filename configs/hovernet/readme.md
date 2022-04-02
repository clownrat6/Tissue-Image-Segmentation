# Horizontal Vertical Network

[HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images](https://arxiv.org/pdf/1812.06499.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--      | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| HoverNet | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 83.07 | 61.14 | 78.24 | 79.71 | 62.36 | 82.7    | 61.31  | 75.78 | 77.92 | 59.3  | 

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | whole      | Adam-Lr1e-4   | 68.81 | 51.48 | 66.02 | 82.77 | 55.14 |

raw res:

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | whole      | Adam-Lr1e-4   | 66.15 | 47.74 | 62.35 | 82.11 | 51.76 |
