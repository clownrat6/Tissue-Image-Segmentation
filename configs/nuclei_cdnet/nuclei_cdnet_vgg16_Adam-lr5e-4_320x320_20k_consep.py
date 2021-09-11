_base_ = [
    '../_base_/datasets/consep_320x320.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
]

# local base
_base_ += [
    './_base_/nuclei_cdnet_vgg16.py',
    './_base_/nuclei_cdnet_hooks.py',
    './_base_/nuclei_cdnet_evaluation.py',
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
