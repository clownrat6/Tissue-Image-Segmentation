# Distance Regression Network

[Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map](https://www.researchgate.net/profile/Peter-Naylor/publication/327065839_Segmentation_of_Nuclei_in_Histopathology_Images_by_Deep_Regression_of_the_Distance_Map/links/5b8412ea92851c1e1235ab7d/Segmentation-of-Nuclei-in-Histopathology-Images-by-Deep-Regression-of-the-Distance-Map.pdf)

## Results

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--    | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| DIST   | 256x256   | 16         | 40x40      | Adam-Lr1e-3   | 59.13 | 80.27 | 73.33  | 73.79 | 54.43 |

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| DIST   | 256x256   | 16         | 40x40      | Adam-Lr1e-3   | 60.17    | 46.52  | 65.87  | 61.12 | 77.43 | 47.52 |
