# dataset settings
dataset_type = 'MoNuSegDataset'
data_root = 'data/monuseg'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1000), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='EdgeMapCalculation'),
    dict(type='InstanceMapCalculation'),
    dict(type='PointMapCalculation'),
    dict(type='DirectionMapCalculation'),
    dict(type='Normalize', max_min=False),
    # dict(type='Standardization', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        data_keys=['img'],
        label_keys=[
            'gt_semantic_map', 'gt_semantic_map_edge', 'gt_point_map',
            'gt_direction_map'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1000),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Standardization', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', data_keys=['img'], label_keys=[]),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/',
        ann_dir='train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/',
        ann_dir='test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/',
        ann_dir='test/',
        pipeline=test_pipeline))
