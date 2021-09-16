# Dataset Prepare

It is recommended to symlink the dataset root to `$ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```None
data
├── consep
│   ├── CoNSeP
│   │    ├── Test
│   │    │   ├── Images
│   │    │   │   └── xxx.png
│   │    │   ├── Labels
│   │    │   │   └── xxx.mat
│   │    │   └── Overlay
│   │    │       └── xxx.png
│   │    └── Train
│   │        ├── Images
│   │        │   └── xxx.png
│   │        ├── Labels
│   │        │   └── xxx.mat
│   │        └── Overlay
│   │            └── xxx.png
│   ├── train
│   ├── test
│   ├── train_c256_s256
│   ├── test_c256_s256 # crop_size = 256 & crop_stride = 256
│   ├── only-train_train.txt
│   ├── only-train_test.txt
│   ├── official_train.txt
│   └── official_test.txt
│
│
├── cpm17
│   ├── CPM17
│   │    ├── test
│   │    │   ├── Images
│   │    │   │   └── xxx.png
│   │    │   ├── Labels
│   │    │   │   └── xxx.mat
│   │    │   └── Overlay
│   │    │       └── xxx.png
│   │    └── train
│   │        ├── Images
│   │        │   └── xxx.png
│   │        ├── Labels
│   │        │   └── xxx.mat
│   │        └── Overlay
│   │            └── xxx.png
│   ├── train
│   ├── test
│   ├── train.txt
│   └── test.txt
│
│
└── monuseg
    ├── MoNuSeg
    │   ├── MoNuSeg 2018 Training Data
    │   │   └── MoNuSeg 2018 Training Data
    │   │       ├── Annotations
    │   │       └── Tissue Images
    │   └── MoNuSegTestData
    │       └── MoNuSegTestData
    │           ├── xxx.tif
    │           └── xxx.xml
    ├── train
    ├── text
    ├── train.txt
    └── test.txt

```

## MoNuSeg Nuclei Segmentation Dataset

1. Download train cohort `"MoNuSeg 2018 Training Data.zip"` and test cohort `"MoNuSegTestData.zip"` from [this](https://monuseg.grand-challenge.org/Data/);
2. Uncompress them into `data/monuseg/MoNuSeg`;
3. If you want to use official dataset split, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg official`;
4. If you want to use train cohort of dataset to split train/test, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg only-train`;
5. If you want to use official split of dataset, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg official`;
6. If you want to use fix-crop dataset, run convertion script:

```python
# only-train split
python tools/convert_dataset/monuseg.py data/monuseg official --crop-size 256 --crop-stride 256
# official split
python tools/convert_dataset/monuseg.py data/monuseg only-train --crop-size 256 --crop-stride 256
```

## CPM17 Nuclei Segmentation Dataset

1. Download cpm17 whole folder from [goole drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK);
2. Put it into `data/cpm17`;
3. Run convertion script: `python tools/convert_dataset/cpm17.py data/cpm17`;

## CoNSeP Nuclei Segmentation Dataset

1. Download CoNSeP dataset from [homepage](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/);
2. Uncompress them into `data/consep`;
3. Run convertion script: `python tools/convert_dataset/consep.py data/consep`;

## Cityscapes Instance Segmentation Dataset
