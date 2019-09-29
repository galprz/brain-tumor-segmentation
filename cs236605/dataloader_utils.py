import torch
from torch.utils.data import DataLoader


def flatten(dataloader: DataLoader):
    """
    Combines batches from a DataLoader into a single tensor. If
    there are multiple tensors returned in each batch, they will be
    flattened separately, returning multiple tensors.

    For example, if the DataLoader returns a samples tensor of shape NxD and a
    labels tensor of shape Nx1 for each batch (N is the batch size),
    this function will return a tuple of two tensors of shapes
    (N*M)xD and (N*M)x1 where M is the number of batches.

    :param dataloader: The DataLoader to flatten.
    :return: A tuple of one or more tensors containing the data from all
        batches.
    """

    out_tensors_cache = []

    for batch in dataloader:

        # Handle case of batch being a tensor (no labels)
        if torch.is_tensor(batch):
            batch = (batch,)
        # Handle case of batch being a dict
        elif isinstance(batch, dict):
            batch = tuple(batch[k] for k in sorted(batch.keys()))
        elif not isinstance(batch, tuple) and not isinstance(batch, list):
            raise TypeError("Unexpected type of batch object")

        for i, tensor in enumerate(batch):
            if i >= len(out_tensors_cache):
                out_tensors_cache.append([])

            out_tensors_cache[i].append(tensor)

    out_tensors = tuple(
        # 0 is batch dimension
        torch.cat(tensors_list, dim=0) for tensors_list in out_tensors_cache
    )

    return out_tensors
