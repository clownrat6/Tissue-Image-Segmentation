_base_ = [
    '../../_base_/datasets/monuseg.py',
    '../../_base_/default_runtime.py',
]

epoch_iter = 12
epoch_num = 400
max_iters = epoch_iter * epoch_num
log_config = dict(interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True), dict(type='TensorboardLoggerHook')])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch_num)

evaluation = dict(
    interval=50,
    custom_intervals=[1],
    custom_milestones=[395],
    by_epoch=True,
    metric='all',
    save_best='mAji',
    rule='greater',
)
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
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
