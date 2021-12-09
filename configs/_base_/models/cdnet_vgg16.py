# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='TorchVGG16BN',
        in_channels=3,
        out_indices=(0, 1, 2, 3, 4),
        pretrained=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
        type='CDHead',
        num_classes=3,
        num_angles=8,
        in_channels=(64, 128, 256, 512, 512),
        in_index=[0, 1, 2, 3, 4],
        stage_convs=[3, 3, 3, 3, 3],
        stage_channels=[64, 128, 256, 512, 512],
        dropout_rate=0.1,
        act_cfg=dict(type='ReLU'),
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
)
