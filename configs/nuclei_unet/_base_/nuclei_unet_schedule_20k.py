_base_ = [
    '../../_base_/schedules/schedule_20k.py',
]
evaluation = dict(
    _delete_=True,
    interval=1000,
    metric='all',
    save_best='Aji',
    rule='greater',
)
checkpoint_config = dict(
    _delete_=True,
    by_epoch=False,
    interval=1000,
    max_keep_ckpts=1,
)
