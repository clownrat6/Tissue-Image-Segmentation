# torch-image-segmentation


## Models and Datasets collection

We collect models and datasets about semantic segmentation and instance segmentation.

Please check [this doc](docs/record.md)

## Dataset Prepare

We can use scripts in `tools/convert_dataset` to complete dataset preparation.

Please check [this doc](docs/dataset_prepare.md)

## Tools

- [ ] Train & Evaluation;
- [ ] Benchmark (inference time, parameters, flops);
- [ ] Dataset (preparation, sample);

Because we use some network ops from mmcv, so the raw version of thop may ignore some ops wrapped by mmcv. We modify the thop to adapt mmcv. mmcv-adapt version thop install command:

```bash
pip install --upgrade git+https://github.com/sennnnn/pytorch-OpCounter.git
```

## Main Board

Support Model:

- [x] 
- [x] 
- [x] 

## Thanks

This repo follow the design mode of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). You see this repo as the son of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).
