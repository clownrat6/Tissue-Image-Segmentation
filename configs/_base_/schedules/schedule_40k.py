_base_ = ['./schedule_10k.py']
runner = dict(max_iters=40000)
checkpoint_config = dict(interval=4000)
evaluation = dict(interval=4000)
