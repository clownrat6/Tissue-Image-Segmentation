# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='Nuclei',
    backbone=dict(
        type='TorchResNet50',
        in_channels=3,
        out_indices=(1, 2, 3, 4),
        pretrained=True,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
    ),
    decode_head=dict(
        type='NucleiUNetHead',
        num_classes=3,
        in_channels=(256, 512, 1024, 2048),
        in_index=[0, 1, 2, 3],
        stage_convs=[3, 3, 3, 3],
        stage_channels=[64, 128, 256, 512],
        dropout_rate=0.1,
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(),
)
