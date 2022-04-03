_base_ = [
    './multi_task_cdnet_vgg16_adam-lr5e-4_bs16_256x256_300e_monuseg_conf.py',
]

# model settings
model = dict(
    type='MultiTaskCDNet',
    # model training and testing settings
    num_classes=2,
    train_cfg=dict(
        num_angles=8,
        use_regression=False,
        noau=False,
        parallel=False,
        use_twobranch=False,
        use_distance=True,
        use_sigmoid=False,
        use_ac=False,
        ac_len_weight=0,
        use_focal=False,
        use_level=False,
        use_variance=False,
        use_tploss=False,
        tploss_weight=False,
        tploss_dice=False,
        dir_weight_map=False,
    ),
    test_cfg=dict(
        mode='split',
        crop_size=(256, 256),
        overlap_size=(40, 40),
        if_ddm=False,
        if_mudslide=False,
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
