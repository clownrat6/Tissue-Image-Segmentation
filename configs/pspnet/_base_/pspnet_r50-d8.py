# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
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
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_rate=0.1,
        num_classes=82,
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        input_transform=None,
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
