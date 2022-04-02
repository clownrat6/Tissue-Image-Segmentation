# Distance Regression Network

[Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map](https://www.researchgate.net/profile/Peter-Naylor/publication/327065839_Segmentation_of_Nuclei_in_Histopathology_Images_by_Deep_Regression_of_the_Distance_Map/links/5b8412ea92851c1e1235ab7d/Segmentation-of-Nuclei-in-Histopathology-Images-by-Deep-Regression-of-the-Distance-Map.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| DIST   | 256x256   | 16         | 40x40      | Adam-Lr1e-3   | 79.75 | 57.34 | 71.68 | 74.61 | 53.49 | 79.44   | 57.55  | 68.36 | 74.08 | 50.83 | 

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| DIST   | 256x256   | 16         | 40x40      | Adam-Lr1e-3   | 65.69 | 46.24 | 61.20 | 77.3  | 47.47 |
