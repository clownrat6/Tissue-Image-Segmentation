_base_ = [
    '../_base_/datasets/carton_oscd.py',
    '../_base_/default_runtime.py',
    '../_base_/models/unet_vgg16.py',
    '../_base_/schedules/mDice_schedule_80k.py',
]

optimizer = dict(_delete_=True, type='Adam', lr=0.001, weight_decay=0.0005)

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
    decode_head=dict(num_classes=3),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

data = dict(samples_per_gpu=8, workers_per_gpu=8)

evaluation = dict(save_best='mDice')
