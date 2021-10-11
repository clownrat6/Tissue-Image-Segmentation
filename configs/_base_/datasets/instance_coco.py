# dataset settings
dataset_type = 'InstanceCOCODataset'
data_root = 'data/coco'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)
# dataset_crop_size = (300, 300)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 320), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=dataset_crop_size, cat_max_ratio=0.75),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='CDNetLabelMake',
        input_level='semantic_with_edge',
        re_edge=True,
        edge_id=81),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='Standardization', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        data_keys=['img'],
        label_keys=[
            'gt_semantic_map_with_edge', 'gt_point_map', 'gt_direction_map'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 320),
        img_ratios=[1.0],
        flip=False,
        flip_direction=['horizontal', 'vertical', 'diagonal'],
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
        img_dir='coco/images/train2017',
        ann_dir='annotations/',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='coco/images/val2017',
        ann_dir='annotations/',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='coco/images/val2017',
        ann_dir='annotations/',
        split='val.txt',
        pipeline=test_pipeline))
