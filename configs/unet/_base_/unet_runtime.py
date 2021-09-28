_base_ = ['../../_base_/default_runtime.py']

custom_hooks = [
    dict(
        type='TrainingCurveHook',
        interval=50,
        plot_keys=[
            'loss',
            'decode.mask_ce_loss',
            'decode.mask_dice_loss',
            'decode.mask_dice',
            'decode.mask_iou',
            'decode.aji',
        ],
        plot_groups=[
            [
                'loss',
            ],
            [
                'decode.mask_ce_loss',
                'decode.mask_dice_loss',
            ],
            [
                'decode.mask_dice',
                'decode.mask_iou',
            ],
            [
                'decode.aji',
            ],
        ],
        axis_groups=[
            [0, 'max_iters', 0, 4],
            [0, 'max_iters', 0, 2],
            [0, 'max_iters', 0, 100],
            [0, 'max_iters', 0, 100],
        ],
        num_rows=2,
        num_cols=2,
    )
]
