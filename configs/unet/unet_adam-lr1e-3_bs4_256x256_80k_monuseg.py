_base_ = [
    '../_base_/datasets/monuseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/Aji_schedule_20k.py',
]

optimizer = dict(_delete_=True, type='Adam', lr=0.0005, weight_decay=0.0005)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# model settings
model = dict(
    type='UNetSegmentor',
    # model training and testing settings
    num_classes=3,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

data = dict(samples_per_gpu=1, workers_per_gpu=1)
