# Deep Contour Aware Network

[DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_DCAN_Deep_Contour-Aware_CVPR_2016_paper.pdf)

## Results

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| DCAN   | 256x256   | 4          | 40x40      | Adam-Lr1e-4   | - | 78.53 | 80.1  | 79.56 | 63.73 |

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| DCAN   | 256x256   | 4          | 40x40      | Adam-Lr1e-4   | 46.66    | 37.45  | 58.95  | 53.66 | 68.01 | 36.43 |
