# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderPostProcess',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='CDHead',
        channels=1,  # useless now
        num_classes=1,  # useless now
        in_channels=(256, 512, 1024, 2048),
        in_index=[0, 1, 2, 3],
        stage_convs=[3, 3, 3, 3],
        stage_channels=[16, 32, 64, 128],
        extra_stage_channels=None,
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
