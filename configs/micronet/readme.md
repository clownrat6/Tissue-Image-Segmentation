# Microscopy Network

[Micro-Net: A unified model for segmentation of various objects in microscopy images](https://www.sciencedirect.com/science/article/abs/pii/S1361841518300628?casa_token=xhKLUM1NLx0AAAAA:vfKqmRR5xqulRzXwXKbeY14ZBxnJUSns1vVDMS8ppEU7zHLnKYVmrk97GW8Hjrn527sYm0xJRMM)

## Results

### MoNuSeg (kumar)

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | Aji   | Dice  | DQ    | SQ    | PQ    |
| :--      | :--:      | :--        | :--:       | :--           | :-:   | :--:  | :--:  | :--:  | :--:  |
| MicroNet | 256x256   | 4          | 40x40      | Adam-Lr1e-4   | 60.3  | 81.63 | 76.7  | 77.42 | 59.38 |

### CoNIC

| Method   | Crop Size | Batch Size | Slide Size | Learning Rate | mean Aji | mAji   | mDice  | mDQ   | mSQ   | mPQ   |
| :--      | :--:      | :--        | :--:       | :--           | :-:      | :--:   | :--:   | :--:  | :--:  | :--:  |
| MicroNet | 256x256   | 4          | 40x40      | Adam-Lr1e-4   | 61.67    | 47.61  | 65.0   | 60.26 | 79.43 | 48.19 |
