import argparse
import json
import os
import os.path as osp
from prettytable import PrettyTable
from collections import OrderedDict


def log_analysis(log_path):
    if osp.isdir(log_path):
        paths = [osp.join(log_path, x) for x in os.listdir(log_path) if osp.splitext(x)[1] == '.json']
        indices = list(range(len(paths)))
        indices = sorted(indices, key=lambda x: osp.getctime(paths[x]), reverse=True)
        paths = [paths[index] for index in indices]
        # The newest path
        log_path = paths[0]

    with open(log_path, 'r') as fp:
        lines = fp.readlines()
        logs = []
        for line in lines:
            line = line.strip()
            logs.append(json.loads(line))

    _ = logs[0]
    logs = logs[1:]

    logs_ = []
    for log in logs:
        if log['mode'] != 'val':
            continue
        logs_.append(log)
    logs_ = logs_[-5:]

    max_mAji = -1
    collect_res = {'mean': {}, 'max': {}}

    inst_keys = ['imwAji', 'bAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ']
    sem_keys = ['mDice']
    dir_keys = ['dir_mDice', 'dir_mPrecision', 'dir_mRecall']
    for log in logs_:
        epoch = log['epoch']
        collect_res[epoch] = {}
        for inst_key in inst_keys:
            collect_res[epoch][inst_key] = log[inst_key]
            if inst_key not in collect_res['mean']:
                collect_res['mean'][inst_key] = [log[inst_key]]
            else:
                collect_res['mean'][inst_key].append(log[inst_key])
        for sem_key in sem_keys:
            collect_res[epoch][sem_key] = log[sem_key]
            if sem_key not in collect_res['mean']:
                collect_res['mean'][sem_key] = [log[sem_key]]
            else:
                collect_res['mean'][sem_key].append(log[sem_key])
        for dir_key in dir_keys:
            if dir_key not in log:
                continue
            collect_res[epoch][dir_key] = log[dir_key]
            if dir_key not in collect_res['mean']:
                collect_res['mean'][dir_key] = [log[dir_key]]
            else:
                collect_res['mean'][dir_key].append(log[dir_key])

        if log['mAji'] > max_mAji:
            collect_res['max'] = collect_res[epoch]
            max_mAji = log['mAji']

    for key in collect_res['mean'].keys():
        collect_res['mean'][key] = round(sum(collect_res['mean'][key]) / len(collect_res['mean'][key]), 2)

    res_table = PrettyTable()
    res = OrderedDict()
    res.update({'names': list(collect_res.keys())})
    res.move_to_end('names', last=False)
    combine_keys = inst_keys + sem_keys + dir_keys
    empty_keys = []
    for key in combine_keys:
        res[key] = []
        for k, single_res in collect_res.items():
            if key not in single_res:
                continue
            res[key].append(single_res[key])
        if len(res[key]) == 0:
            empty_keys.append(key)
    # remove empty key
    [res.pop(key) for key in empty_keys]

    for key, val in res.items():
        res_table.add_column(key, val)

    return res_table.get_string()


def parse_args():
    parser = argparse.ArgumentParser(description='Extract related results.')
    parser.add_argument('log_path', help='The json style log file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    log_path = args.log_path

    print(log_analysis(log_path))


if __name__ == '__main__':
    main()
