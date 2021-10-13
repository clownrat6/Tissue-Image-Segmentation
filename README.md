# torch-image-segmentation

## Dataset Prepare

Please check [this doc](docs/dataset_prepare.md)

Supported Dataset:

- [x] MoNuSeg;
- [x] CoNSeP;
- [x] CPM17;
- [x] OSCD;
- [x] COCO;

## Usage

```Bash
# single gpu training
python tools/train.py [config_path]
# multi gpu training
./tools/dist_train.sh [config_path] [gpu_num]
# demo (cdnet for oscd dataset on 1 gpu)
python tools/train.py configs/cdnet/cdnet_vgg16_adam-lr1e-3_8x1_256x256_80k_carton_oscd.py
# demo (upernet for coco dataset on 4 gpu)
./tools/dist_train.py configs/upernet/upernet_r50-d8_sgd-lr5e-3_4x2_512x512_160k_instance_coco.py 4
```

## Main Board

Support Model:

- [x] UNet
- [x] UPerNet
- [x] CDNet

## Thanks

This repo follow the design mode of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). You see this repo as the son of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
