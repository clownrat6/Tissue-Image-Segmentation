_base_ = ['./schedule_10k.py']
runner = dict(max_iters=20000)
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000)
