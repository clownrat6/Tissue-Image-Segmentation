# Deep Contour Aware Network

[DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation](https://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_DCAN_Deep_Contour-Aware_CVPR_2016_paper.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  |
| DCAN   | 256x256   | 4          | 40x40      | Adam-Lr1e-4   | 78.41 | 47.65 | 67.78 | 71.99 | 48.80 | 77.98   | 52.22  | 65.91 | 71.80 | 47.44 | 

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| DCAN   | 256x256   | 4          | whole      | Adam-Lr1e-4   | 56.28 | 35.6  | 49.9  | 67.02 | 33.44 |
