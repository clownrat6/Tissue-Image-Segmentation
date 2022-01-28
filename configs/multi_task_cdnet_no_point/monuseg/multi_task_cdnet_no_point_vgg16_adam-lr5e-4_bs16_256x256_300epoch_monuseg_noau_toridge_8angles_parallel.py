_base_ = [
    '../../_base_/datasets/monuseg_w_dir.py',
    '../../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'NucleiMoNuSegDatasetWithDirection'
data_root = 'data/monuseg'
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
    test_with_dir=True,
    min_size=256,
    max_size=2048,
    resize_mode='fix',
    edge_id=2,
    to_center=False,
    num_angles=num_angles,
)
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c300',
        ann_dir='train/c300',
        split='only-train_t12_v4_train_c300.txt',
        process_cfg=process_cfg),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c0',
        ann_dir='train/c0',
        split='only-train_t12_v4_test_c0.txt',
        process_cfg=process_cfg),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/c0',
        ann_dir='train/c0',
        split='only-train_t12_v4_test_c0.txt',
        process_cfg=process_cfg),
)

epoch_iter = 12
epoch_num = 300
max_iters = epoch_iter * epoch_num
log_config = dict(
    interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True),
                                dict(type='TensorboardLoggerHook')])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch_num)

evaluation = dict(
    interval=50,
    custom_intervals=[1],
    custom_milestones=[epoch_num-5],
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
    type='MultiTaskCDNetSegmentorNoPoint',
    # model training and testing settings
    num_classes=2,
    train_cfg=dict(if_weighted_loss=False, noau=True, num_angles=num_angles, parallel=True),
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
