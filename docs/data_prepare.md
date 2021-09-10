# Dataset Prepare

It is recommended to symlink the dataset root to `$ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```None
data
├── cpm17
│   ├── backup
│   ├── cpm17
│   │   ├── test
│   │   └── train
│   ├── test
│   ├── test.txt
│   ├── train
│   └── train.txt
└── monuseg
    ├── MoNuSeg 2018 Training Data
    ├── MoNuSegTestData
    ├── test
    ├── test.txt
    ├── train
    └── train.txt
```

## MoNuSeg Nuclei Segmentation Dataset

1. Download train cohort `"MoNuSeg 2018 Training Data.zip"` and test cohort `"MoNuSegTestData.zip"` from [this](https://monuseg.grand-challenge.org/Data/);
2. Uncompress them into `data/monuseg`;
3. Run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg`;

## CPM17 Nuclei Segmentation Dataset

1. Download cpm17 whole folder from [goole drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK);
2. Put it into `data/cpm17`;
3. Run convertion script: `python tools/convert_dataset/cpm17.py data/cpm17`;

## CoNSep Nuclei Segmentation Dataset

## Cityscapes Instance Segmentation Dataset
