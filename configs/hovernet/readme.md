# Horizontal Vertical Network

[HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images](https://arxiv.org/pdf/1812.06499.pdf)

## Results

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--      | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 61.86 | 83.44 | 78.57 | 79.73 | 62.64 |

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| HoverNet | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 65.96    | 51.63  | 68.95  | 66.32 | 82.7  | 55.33 |
