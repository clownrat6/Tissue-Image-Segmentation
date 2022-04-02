# Microscopy Network

[Micro-Net: A unified model for segmentation of various objects in microscopy images](https://www.sciencedirect.com/science/article/abs/pii/S1361841518300628?casa_token=xhKLUM1NLx0AAAAA:vfKqmRR5xqulRzXwXKbeY14ZBxnJUSns1vVDMS8ppEU7zHLnKYVmrk97GW8Hjrn527sYm0xJRMM)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--      | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| MicroNet | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 81.98 | 55.43 | 75.37 | 78.46 | 59.14 | 81.72   | 56.29  | 71.51 | 77.69 | 55.68 | 

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| MicroNet | 252x252   | 4          | 40x40      | Adam-Lr1e-4   | 64.88 | 47.39 | 60.25 | 79.23 | 48.03 |
