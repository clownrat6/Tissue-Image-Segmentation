_base_ = ['./schedule_10k.py']
runner = dict(max_iters=160000)
checkpoint_config = dict(interval=16000)
evaluation = dict(interval=16000)
