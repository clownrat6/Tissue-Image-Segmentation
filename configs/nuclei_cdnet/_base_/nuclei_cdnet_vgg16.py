# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='NucleiCDNet',
    pretrained='mmcls://vgg16_bn',
    backbone=dict(
        type='VGG',
        depth=16,
        in_channels=3,
        base_channels=64,
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        type='NucleiCDHead',
        in_channels=(64, 128, 256, 512, 512),
        in_index=[0, 1, 2, 3, 4],
        stage_convs=[3, 3, 3, 3, 3],
        stage_channels=[64, 128, 256, 512, 512],
        extra_stage_channels=None,
        act_cfg=dict(type='ReLU'),
        norm_cfg=norm_cfg,
        align_corners=False),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(320, 320), stride=(231, 231)))
