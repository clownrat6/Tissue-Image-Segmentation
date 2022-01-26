_base_ = [
    '../../_base_/datasets/conic_w_dir.py',
    '../../_base_/default_runtime.py',
]

# datasets settings
dataset_type = 'NucleiCoNICDataset'
data_root = 'data/conic'
num_angles = 8
process_cfg = dict(
    if_flip=True,
    if_jitter=True,
    if_elastic=True,
    if_blur=True,
    if_crop=True,
    if_pad=True,
    if_norm=False,
    with_dir=True,
    min_size=256,
    max_size=2048,
    resize_mode='fix',
    edge_id=7,
    to_center=False,
    num_angles=num_angles,
)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/',
        ann_dir='train/',
        split='train.txt',
        process_cfg=process_cfg),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/',
        ann_dir='val/',
        split='val.txt',
        process_cfg=process_cfg),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/',
        ann_dir='val/',
        split='val.txt',
        process_cfg=process_cfg),
)

epoch_iter = 247
epoch_num = 200
max_iters = epoch_iter * epoch_num
log_config = dict(interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True), dict(type='TensorboardLoggerHook')])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=200)

evaluation = dict(
    interval=50,
    custom_intervals=[1],
    custom_milestones=[195],
    by_epoch=True,
    metric='all',
    save_best='Aji',
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
    num_classes=7,
    train_cfg=dict(use_ac=True, ac_len_weight=0),
    test_cfg=dict(
        mode='whole',
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
