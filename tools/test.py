import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from tiseg.apis import multi_gpu_test, single_gpu_test
from tiseg.datasets import build_dataloader, build_dataset
from tiseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--work-dir', type=str, help='The dump folder of evaluation results')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.eval or args.format_only, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = True
        cfg.data.test.pipeline[1].rotate = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not isinstance(cfg.data.test, list):
        cfg.data.test = [cfg.data.test]

    data_test_list = cfg.data.test
    for data_test in data_test_list:
        data_test.test_mode = True

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(data_test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE

        eval_kwargs = {} if args.eval_options is None else args.eval_options

        eval_kwargs['dump_path'] = (None if args.work_dir is None else
                                    f'{args.work_dir}/evaluation_dump.txt')

        if args.format_only:
            if 'imgfile_prefix' in eval_kwargs:
                tmpdir = eval_kwargs['imgfile_prefix']
            else:
                tmpdir = '.format_temp'
                eval_kwargs.setdefault('imgfile_prefix', tmpdir)
            mmcv.mkdir_or_exist(tmpdir)
        else:
            tmpdir = None

        # clean gpu memory when starting a new evaluation.
        torch.cuda.empty_cache()

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            results = single_gpu_test(
                model,
                data_loader,
                pre_eval=args.eval is not None,
                format_only=args.format_only,
                format_args=eval_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            results = multi_gpu_test(
                model,
                data_loader,
                args.gpu_collect,
                pre_eval=args.eval is not None,
                format_only=args.format_only,
                format_args=eval_kwargs)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.eval:
                dataset.evaluate(results, args.eval, **eval_kwargs)


if __name__ == '__main__':
    main()
