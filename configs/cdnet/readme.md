# Centripetal Direction Network

[CDNet: Centripetal Direction Network for Nuclear Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/He_CDNet_Centripetal_Direction_Network_for_Nuclear_Instance_Segmentation_ICCV_2021_paper.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| CDNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 82.60 | 62.98 | 80.24 | 79.05 | 63.43 | 82.04   | 62.39  | 77.30 | 77.88 | 60.34 | 

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| CDNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 62.4  | 44.72 | 61.37 | 73.28 | 45.05 |


## Experiments

### Boundary Width

| Method | Dilation | Erosion    | Bound Width | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--      | :--        | :--:        | :-:   | :--:  | :--:  | :--:  | :--:  |
| CDNet  | 0        | 2          | 2           | 62.23 | 83.53 | 79.57 | 79.75 | 63.46 |
| CDNet  | 0        | 3          | 3           | 62.87 | 83.5  | 81.32 | 80.35 | 65.34 |
| CDNet  | 3        | 3          | 6           | 63.73 | 83.59 | 80.1  | 79.56 | 63.73 |
