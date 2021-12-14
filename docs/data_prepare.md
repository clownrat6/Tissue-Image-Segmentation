# Dataset Prepare

It is recommended to symlink the dataset root to `$ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```None
data
|
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
│   ├── train.txt
│   └── test.txt
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
|
├── oscd
│   ├── OSCD
|   │   ├── coco_carton
|   │   └── labelme
│   ├── annotations
│   ├── images
│   ├── train.txt
│   └── test.txt
│
└── monuseg
    ├── MoNuSeg
    │   ├── MoNuSeg 2018 Training Data
    │       ├── Annotations
    │   │   └── Tissue Images
    │   └── MoNuSegTestData
    │       ├── xxx.tif
    │       └── xxx.xml
    ├── train
    ├── test
    ├── only-train_t12_v4_train_c300.txt
    ├── only-train_t12_v4_test_c0.txt
    └── only-train_t12_v4_val_c0.txt

```

## MoNuSeg Nuclei Segmentation Dataset

1. Download train cohort `"MoNuSeg 2018 Training Data.zip"` and test cohort `"MoNuSegTestData.zip"` from [this](https://monuseg.grand-challenge.org/Data/);
2. Uncompress them into `data/monuseg/monuseg`;
3. If you want to use our dataset split, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg only-train_t12_v4`;
6. If you want to use fix-crop dataset, run convertion script:

```python
# only-train_t12_v4 (our split) crop_size = 300x300
python tools/convert_dataset/monuseg.py data/monuseg only-train_t12_v4 -c 300
```

## CPM17 Nuclei Segmentation Dataset

1. Download cpm17 whole folder from [goole drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK);
2. Put it into `data/cpm17/CPM17`;
3. Run convertion script: `python tools/convert_dataset/cpm17.py data/cpm17 -c 300`;

## CoNSeP Nuclei Segmentation Dataset

1. Download CoNSeP dataset from [homepage](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/);
2. Uncompress them into `data/consep`;
3. Run convertion script: `python tools/convert_dataset/consep.py data/consep -c 300`;

## OCSD Carton Segmentation Dataset

1. Download OCSD from [homepage](https://github.com/yancie-yjr/scd.github.io);
2. Get `OSCD.zip` from [BaiduNetDisk](https://pan.baidu.com/s/1p2KOYFhLWFfbmMBLpxbVMA);
3. Uncompressed `OSCD.zip` into `data/oscd`
4. Run conversion script: `python tools/convert_dataset/oscd.py data/oscd --nproc 8`;
