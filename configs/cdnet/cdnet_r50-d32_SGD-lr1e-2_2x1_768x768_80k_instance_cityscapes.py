_base_ = [
    '../_base_/datasets/instance_cityscapes.py',
    './_base_/cdnet_runtime.py',
    './_base_/cdnet_r50-d32.py',
    './_base_/cdnet_schedule_20k.py',
]

optimizer = dict(_delete_=True, type='SGD', lr=0.01, weight_decay=0.0005)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

model = dict(
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide', crop_size=(768, 768), stride=(513, 513), use_ddm=True),
)

data = dict(samples_per_gpu=2, workers_per_gpu=2)
