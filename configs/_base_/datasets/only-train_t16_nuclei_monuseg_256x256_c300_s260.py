# dataset settings
dataset_type = 'NucleiMoNuSegDataset'
data_root = 'data/monuseg'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='CDNetLabelMake', input_level='semantic_with_edge', re_edge=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Normalize', max_min=False),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        data_keys=['img'],
        label_keys=[
            'gt_semantic_map', 'gt_semantic_map_with_edge', 'gt_point_map',
            'gt_direction_map'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1000),
        img_ratios=[1.0],
        flip=True,
        flip_direction=['horizontal', 'vertical', 'diagonal'],
        rotate=True,
        rotate_degree=[90],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='RandomSparseRotate'),
            dict(type='Normalize', max_min=False),
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
        img_dir='train_c300_s260/',
        ann_dir='train_c300_s260/',
        split='only-train_t16_train_c300_s260.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/',
        ann_dir='train/',
        split='only-train_t16_test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/',
        ann_dir='train/',
        split='only-train_t16_test.txt',
        pipeline=test_pipeline))
