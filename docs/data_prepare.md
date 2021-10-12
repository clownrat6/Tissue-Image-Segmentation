# Dataset Prepare

It is recommended to symlink the dataset root to `$ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```None
data
├── coco
│   ├── coco
│   │    ├── images
│   │    │   ├── train2017
│   │    │   └── val2017
│   ├── annotations
│   ├── train.txt
│   └── val.txt
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
    ├── test
    ├── train_c256_s256
    ├── test_c256_s256 # crop_size = 256 & crop_stride = 256
    ├── only-train_train.txt
    ├── only-train_test.txt
    ├── only-train_train_c256_s256.txt
    ├── only-train_test_c256_s256.txt
    ├── official_train.txt
    └── official_test.txt

```

## MoNuSeg Nuclei Segmentation Dataset

1. Download train cohort `"MoNuSeg 2018 Training Data.zip"` and test cohort `"MoNuSegTestData.zip"` from [this](https://monuseg.grand-challenge.org/Data/);
2. Uncompress them into `data/monuseg/MoNuSeg`;
3. If you want to use official dataset split, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg official`;
4. If you want to use train cohort of dataset to split train/test, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg only-train`;
5. If you want to use official split of dataset, run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg official`;
6. If you want to use fix-crop dataset, run convertion script:

```python
# only-train split (crop_size=256x256, stride=256)
python tools/convert_dataset/monuseg.py data/monuseg only-train --crop-size 256 --crop-stride 256
# official split (crop_size=256x256, stride=256)
python tools/convert_dataset/monuseg.py data/monuseg official --crop-size 256 --crop-stride 256
# official split (whole)
python tools/convert_dataset/monuseg.py data/monuseg official
```

## CPM17 Nuclei Segmentation Dataset

1. Download cpm17 whole folder from [goole drive](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK);
2. Put it into `data/cpm17/CPM17`;
3. Run convertion script: `python tools/convert_dataset/cpm17.py data/cpm17`;

## CoNSeP Nuclei Segmentation Dataset

1. Download CoNSeP dataset from [homepage](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/);
2. Uncompress them into `data/consep`;
3. Run convertion script: `python tools/convert_dataset/consep.py data/consep`;

## COCO Instance Segmentation Dataset

1. Download COCO dataset from [homepage](https://cocodataset.org/#download);
2. Download [2017 train images](http://images.cocodataset.org/zips/train2017.zip) & [2017 val images](http://images.cocodataset.org/zips/val2017.zip);
3. Uncompressed them into `data/coco/coco/images`;
4. Download [2017 train/val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip);
5. Uncompressed it into `data/coco/coco/annotations`;
6. Run conversion script: `python tools/convert_dataset/coco.py data/coco --nproc 8`;
