_base_ = [
    '../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'NucleiMoNuSegDataset'
data_root = 'data/monuseg'
train_processes = [
    dict(type='Affine', scale=(0.8, 1.2), shear=5, rotate_degree=[-180, 180], translate_frac=(0, 0.01)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='Pad', pad_size=(256, 256)),
    dict(type='RandomBlur'),
    dict(
        type='ColorJitter', hue_delta=8, saturation_range=(0.8, 1.2), brightness_delta=26, contrast_range=(0.75, 1.25)),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='BoundLabelMake', edge_id=2, selem_radius=(0, 2)),
    dict(type='Formatting', data_keys=['img'], label_keys=['sem_gt', 'sem_gt_w_bound']),
]
test_processes = [
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='Formatting', data_keys=['img'], label_keys=[]),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/w512_s256',
        ann_dir='train/w512_s256',
        split='only-train_t12_v4_train_w512_s256.txt',
        processes=train_processes),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/w0_s0',
        ann_dir='train/w0_s0',
        split='only-train_t12_v4_test_w0_s0.txt',
        processes=test_processes),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/w0_s0',
        ann_dir='train/w0_s0',
        split='only-train_t12_v4_test_w0_s0.txt',
        processes=test_processes),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)

evaluation = dict(
    interval=20,
    custom_intervals=[1],
    custom_milestones=[295],
    by_epoch=True,
    metric='all',
    save_best='Aji',
    rule='greater',
)

checkpoint_config = dict(
    by_epoch=True,
    interval=5,
    max_keep_ckpts=5,
)

optimizer = dict(type='Adam', lr=0.0005, weight_decay=0.0005)
optimizer_config = dict()

# NOTE: poly learning rate decay
# lr_config = dict(
#     policy='poly', warmup='linear', warmup_iters=100, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)

# NOTE: fixed learning rate decay
# lr_config = dict(policy='fixed', warmup='linear', warmup_iters=100, warmup_ratio=1e-6, by_epoch=False)

# NOTE: step learning rate decay
lr_config = dict(
    policy='step', by_epoch=True, step=[200], gamma=0.1, warmup='linear', warmup_iters=100, warmup_ratio=1e-6)

# model settings
model = dict(
    type='UNet',
    # model training and testing settings
    num_classes=2,
    train_cfg=dict(),
    test_cfg=dict(
        mode='split',
        radius=1,
        crop_size=(256, 256),
        overlap_size=(40, 40),
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
