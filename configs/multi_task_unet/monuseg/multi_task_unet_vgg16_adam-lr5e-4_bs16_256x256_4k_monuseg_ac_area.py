_base_ = [
    '../../_base_/datasets/monuseg.py',
    '../../_base_/default_runtime.py',
]

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=4000)

evaluation = dict(
    interval=200,
    eval_start=0,
    metric='all',
    save_best='mAji',
    rule='greater',
)
checkpoint_config = dict(
    by_epoch=False,
    interval=200,
    max_keep_ckpts=1,
)

optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0005)
optimizer_config = dict()

# NOTE: poly learning rate decay
# lr_config = dict(
#     policy='poly', warmup='linear', warmup_iters=100, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)

# NOTE: fixed learning rate decay
lr_config = dict(policy='fixed', warmup=None, warmup_iters=100, warmup_ratio=1e-6, by_epoch=False)

# model settings
model = dict(
    type='MultiTaskUNetSegmentor',
    # model training and testing settings
    num_classes=2,
    train_cfg=dict(use_ac=True, ac_w_area=True),
    test_cfg=dict(
        mode='whole',
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
