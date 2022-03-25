# Centripetal Direction Network

[CDNet: Centripetal Direction Network for Nuclear Instance Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/He_CDNet_Centripetal_Direction_Network_for_Nuclear_Instance_Segmentation_ICCV_2021_paper.pdf)

## Results

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| CDNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 63.73 | 83.59 | 80.1  | 79.56 | 63.73 |

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| CUNet  | 256x256   | 16         | 40x40      | Adam-Lr5e-4   | 58.51    | 44.86  | 62.59  | 61.65 | 73.14 | 45.13 |

## Experiments

### Boundary Width

| Method | Dilation | Erosion    | Bound Width | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--      | :--        | :--:        | :-:   | :--:  | :--:  | :--:  | :--:  |
| CUNet  | 0        | 2          | 2           | 62.23 | 83.53 | 79.57 | 79.75 | 63.46 |
| CUNet  | 0        | 3          | 3           | 62.87 | 83.5  | 81.32 | 80.35 | 65.34 |
| CUNet  | 3        | 3          | 6           | 63.73 | 83.59 | 80.1  | 79.56 | 63.73 |
