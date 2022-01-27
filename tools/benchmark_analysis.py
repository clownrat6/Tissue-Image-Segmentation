import argparse
import os
import os.path as osp
import pickle as p
from prettytable import PrettyTable
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='test (and eval) a model')
    parser.add_argument('analysis_folder', help='The analysis log save folder.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    analysis_folder = args.analysis_folder

    analysis_logs = [osp.join(analysis_folder, x) for x in os.listdir(analysis_folder) if osp.splitext(x)[1] == '.p']

    max_mAji = -1
    collect_res = {'max': {}, 'mean': {}}

    inst_keys = ['imwAji', 'bAji', 'mAji', 'bDQ', 'bSQ', 'bPQ', 'imwDQ', 'imwSQ', 'imwPQ', 'mDQ', 'mSQ', 'mPQ']
    sem_keys = ['mDice']
    for analysis_log in analysis_logs:
        log_name = osp.splitext(osp.basename(analysis_log))[0]
        log_dict = p.load(open(analysis_log, 'rb'))

        inst_metrics = log_dict['total_inst_metrics']
        sem_metrics = log_dict['total_sem_metrics']

        collect_res[log_name] = {}
        for inst_key in inst_keys:
            collect_res[log_name][inst_key] = inst_metrics[inst_key]
            if inst_key not in collect_res['mean']:
                collect_res['mean'][inst_key] = [inst_metrics[inst_key]]
            else:
                collect_res['mean'][inst_key].append(inst_metrics[inst_key])
        for sem_key in sem_keys:
            collect_res[log_name][sem_key] = sem_metrics[sem_key]
            if sem_key not in collect_res['mean']:
                collect_res['mean'][sem_key] = [sem_metrics[sem_key]]
            else:
                collect_res['mean'][sem_key].append(sem_metrics[sem_key])
        if inst_metrics['mAji'] > max_mAji:
            collect_res['max'] = collect_res[log_name]

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