_base_ = [
    '../../_base_/schedules/schedule_20k.py',
]
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric='all',
    save_best='aji',
    rule='greater',
)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=1,
)
