_base_ = [
    './_base_/nuclei_cdnet_r50-d8.py', '../_base_/datasets/monuseg_320x320.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

optimizer = dict(_delete_=True, type='Adam', lr=0.00001, weight_decay=0.0005)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)