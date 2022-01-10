import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np
import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash, collect_env, get_logger

from tiseg import __version__
from tiseg.apis import train_segmentor
from tiseg.datasets import build_dataset
from tiseg.models import build_segmentor
from tiseg.models.utils import revert_sync_batchnorm


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse Log')
    parser.add_argument('log_path', help='log file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path = args.log_path
    save_dir = osp.dirname(path)
    # print(path)
    # print(save_dir)
    results = []
    with open(path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            # print(lines)
            if not lines:
                break
            if ("imwAji |" in lines):
                for i in range(2):
                    lines = file_to_read.readline()
                results.append(lines[:-1])
                # print(lines[:-1])
    all_ans = []
    for res in results[-5:]:
        print(res)
        ans = []
        l = 0
        cnt = 0
        while True:
            cnt += 1
            r = res[l+1:].find('|')
            if (r == -1):
                break
            r = r + l + 1
            ans.append(float(res[l+1:r]))
            l = r
        all_ans.append(ans)
    ans = np.array(all_ans)
    ans = np.mean(ans, axis=0)
    print(ans)
        # print(res.find('|'))
if __name__ == '__main__':
    main()
