# Data Preparetion

## MoNuSeg Nuclei Segmentation Dataset

1. Download train cohort `"MoNuSeg 2018 Training Data.zip"` and test cohort `"MoNuSegTestData.zip"` from [this](https://monuseg.grand-challenge.org/Data/);
2. Uncompress them into `data/monuseg`;
3. Run convertion script: `python tools/convert_dataset/monuseg.py data/monuseg --with_edge`;
