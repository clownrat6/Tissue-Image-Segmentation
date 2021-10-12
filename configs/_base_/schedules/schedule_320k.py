_base_ = ['./schedule_10k.py']
runner = dict(max_iters=320000)
checkpoint_config = dict(interval=32000)
evaluation = dict(interval=32000)
