# U-Net

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

## Results

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--      | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| UNet     | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 59.12 | 82.34 | 74.46 | 78.07 | 58.14 |

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| UNet     | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 60.88    | 49.26  | 67.6   | 62.24 | 79.3  | 49.6  |
