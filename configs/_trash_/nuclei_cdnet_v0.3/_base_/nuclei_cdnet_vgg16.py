# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='NucleiCDNet',
    backbone=dict(
        type='TorchVGG16BN',
        in_channels=3,
        out_indices=(0, 1, 2, 3, 4),
        pretrained=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
        type='NucleiCDHead',
        dropout_rate=0.1,
        in_channels=(64, 128, 256, 512, 512),
        in_index=[0, 1, 2, 3, 4],
        stage_convs=[3, 3, 3, 3, 3],
        stage_channels=[32, 64, 128, 256, 256],
        extra_stage_channels=None,
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict())
