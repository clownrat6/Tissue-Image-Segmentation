import mmcv
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.runner import get_dist_info


def single_gpu_test(model,
                    data_loader,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    pre_eval_args={}):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        opacity(float): Opacity of painted segmentation map, Must be in (0, 1]
            range. Default 0.5.

    Returns:
        object: The processor containing results.
    """
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [pre_eval, format_only].count(True) <= 1, \
        '``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset

    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(
                result, indices=batch_indices, **pre_eval_args)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model,
                   data_loader,
                   gpu_collect=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={},
                   pre_eval_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """

    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [pre_eval, format_only].count(True) <= 1, \
        '``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(
                result, indices=batch_indices, **pre_eval_args)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), None)
    return results
