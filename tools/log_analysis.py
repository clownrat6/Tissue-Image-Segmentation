import argparse
import json
from prettytable import PrettyTable
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Analyse Log')
    parser.add_argument('log_path', help='log file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.log_path, 'r') as fp:
        lines = fp.readlines()
        logs = []
        for line in lines:
            line = line.strip()
            logs.append(json.loads(line))

    info = logs[0]
    logs = logs[1:]

    logs_ = []
    for log in logs:
        if log['mode'] != 'val':
            continue
        logs_.append(log)
    logs_ = logs_[-5:]

    max_mAji = -1
    collect_res = {'max': {}, 'mean': {}}

    inst_keys = ['imwAji', 'bAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ']
    sem_keys = ['mDice']

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
        if log['mAji'] > max_mAji:
            collect_res['max'] = collect_res[epoch]

    for key in collect_res['mean'].keys():
        collect_res['mean'][key] = round(sum(collect_res['mean'][key]) / len(collect_res['mean'][key]), 2)

    res_table = PrettyTable()
    res = OrderedDict()
    res.update({'names': list(collect_res.keys())})
    res.move_to_end('names', last=False)
    combine_keys = inst_keys + sem_keys
    for key in combine_keys:
        res[key] = []
        for k, single_res in collect_res.items():
            res[key].append(single_res[key])

    for key, val in res.items():
        res_table.add_column(key, val)

    print(res_table.get_string())


if __name__ == '__main__':
    main()