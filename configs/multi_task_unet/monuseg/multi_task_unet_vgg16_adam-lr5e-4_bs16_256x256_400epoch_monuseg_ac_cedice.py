_base_ = [
    '../../_base_/datasets/monuseg.py',
    '../../_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'NucleiMoNuSegDatasetWithDirection'
data_root = 'data/monuseg'
process_cfg = dict(
    if_flip=True,
    if_jitter=True,
    if_elastic=True,
    if_blur=True,
    if_crop=True,
    if_pad=True,
    if_norm=False,
<<<<<<< HEAD
    with_dir=True,
    test_with_dir=True,
=======
<<<<<<< HEAD:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_200epoch_monuseg.py
    with_dir=False,
=======
    with_dir=True,
    test_with_dir=True,
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_400epoch_monuseg_ac_cedice.py
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
    min_size=256,
    max_size=2048,
    resize_mode='fix',
    edge_id=2,
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
<<<<<<< HEAD
epoch_num = 400
max_iters = epoch_iter * epoch_num
log_config = dict(
    interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True),
                                dict(type='TensorboardLoggerHook')])
=======
epoch_num = 200
max_iters = epoch_iter * epoch_num
<<<<<<< HEAD:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_200epoch_monuseg.py
log_config = dict(interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True), dict(type='TensorboardLoggerHook')])
=======
log_config = dict(
    interval=epoch_iter, hooks=[dict(type='TextLoggerHook', by_epoch=True),
                                dict(type='TensorboardLoggerHook')])
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_400epoch_monuseg_ac_cedice.py
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epoch_num)

evaluation = dict(
    interval=50,
    custom_intervals=[1],
<<<<<<< HEAD
    custom_milestones=[395],
=======
<<<<<<< HEAD:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_200epoch_monuseg.py
    custom_milestones=[390],
=======
    custom_milestones=[395],
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_400epoch_monuseg_ac_cedice.py
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
    by_epoch=True,
    metric='all',
    save_best='mAji',
    rule='greater',
)
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
<<<<<<< HEAD
    max_keep_ckpts=5,
)

=======
<<<<<<< HEAD:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_200epoch_monuseg.py
    max_keep_ckpts=10,
=======
    max_keep_ckpts=5,
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_400epoch_monuseg_ac_cedice.py
)



>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
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
<<<<<<< HEAD
    train_cfg=dict(use_ac=True, ac_len_weight=0),
=======
<<<<<<< HEAD:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_200epoch_monuseg.py
    train_cfg=dict(),
=======
    train_cfg=dict(use_ac=True, ac_len_weight=0),
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299:configs/multi_task_unet/monuseg/multi_task_unet_vgg16_adam-lr5e-4_bs16_256x256_400epoch_monuseg_ac_cedice.py
>>>>>>> 542611ab99376da15c29c65944d0f8ab7816a299
    test_cfg=dict(
        mode='whole',
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
