_base_ = [
    '../_base_/datasets/monuseg.py',
    '../_base_/default_runtime.py',
]

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)

evaluation = dict(
    _delete_=True,
    interval=1000,
    metric='all',
    save_best='Aji',
    rule='greater',
)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=1000,
    max_keep_ckpts=1,
)

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
