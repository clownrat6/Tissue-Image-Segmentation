_base_ = [
    '../../_base_/schedules/schedule_80k.py',
]
evaluation = dict(
    _delete_=True,
    interval=4000,
    metric='all',
    save_best='mIoU',
    rule='greater',
)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=4000,
    max_keep_ckpts=1,
)
