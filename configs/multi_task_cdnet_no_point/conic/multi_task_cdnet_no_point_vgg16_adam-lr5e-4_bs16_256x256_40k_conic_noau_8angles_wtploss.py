_base_ = [
    '../../_base_/datasets/conic_w_dir.py',
    '../../_base_/default_runtime.py',
]

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)

evaluation = dict(
    interval=1000,
    metric='all',
    save_best='mAji',
    rule='greater',
)
checkpoint_config = dict(
    by_epoch=False,
    interval=1000,
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
    type='MultiTaskCDNetSegmentorNoPoint',
    # model training and testing settings
    num_classes=7,
    train_cfg=dict(
        if_weighted_loss=False, 
        noau=True, 
        use_tploss=True,
        tploss_weight=True,
    ),
    test_cfg=dict(
        mode='split',
        plane_size=(256, 256),
        crop_size=(256, 256),
        overlap_size=(80, 80),
        if_ddm=False,
        if_mudslide=False,
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
