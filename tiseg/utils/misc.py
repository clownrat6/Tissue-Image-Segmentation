import numpy as np
import torch


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def tensor2maps(tensor):
    """Convert tensor to ground truth instance map.

    Args:
        tensor (Tensor): Tensor that contains multiple ground truth instance
            maps, shape (N, C, H, W).

    Returns:
        list[np.ndarray]: A list that contains multiple ground truth instance
            maps.
    """
    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 4

    num_maps = tensor.size(0)
    maps = []
    for map_id in range(num_maps):
        map = tensor[map_id, ...].cpu().numpy()[0, :, :].astype(np.uint8)
        maps.append(np.ascontiguousarray(map))
    return maps


def tensor2txts(tensor):
    """Convert tensor to txt indices vector.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple texts, shape (
            N, C, L).

    Returns:
        list[np.ndarray]: A list that contains multiple texts.
    """
    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and (tensor.ndim == 3 or tensor.ndim == 2)
    if tensor.ndim == 3:
        tensor = tensor[:, 0, :]
    num_txts = tensor.size(0)
    txts = []
    for txt_id in range(num_txts):
        txt = tensor[txt_id].cpu().numpy()
        txts.append(np.ascontiguousarray(txt))
    return txts
