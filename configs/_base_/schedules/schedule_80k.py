_base_ = ['./schedule_10k.py']
runner = dict(max_iters=80000)
checkpoint_config = dict(interval=8000)
evaluation = dict(interval=8000)
