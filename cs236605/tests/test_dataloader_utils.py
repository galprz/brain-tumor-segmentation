import pytest

import torch
from torch.utils.data import DataLoader, Dataset
import cs236605.dataloader_utils as dl_utils


DATASET_SIZE = 1000
DATA_SIZE = 100


class TestFlatten(object):

    def test_tensor(self):
        loader = DataLoader(TensorDataset(), batch_size=256)

        x, = dl_utils.flatten(loader)

        assert torch.is_tensor(x)
        assert x.shape == torch.Size([DATASET_SIZE, DATA_SIZE, DATA_SIZE])

    def test_two_tuple(self):
        loader = DataLoader(TensorTwoTupleDataset(), batch_size=256)

        x, y = dl_utils.flatten(loader)

        assert torch.is_tensor(x)
        assert torch.is_tensor(y)
        assert x.shape == torch.Size([DATASET_SIZE, DATA_SIZE, DATA_SIZE])
        assert y.shape == torch.Size([DATASET_SIZE, DATA_SIZE, 1])

    def test_three_tuple(self):
        loader = DataLoader(TensorThreeTupleDataset(), batch_size=128)

        x, y, z = dl_utils.flatten(loader)

        assert torch.is_tensor(x)
        assert torch.is_tensor(y)
        assert torch.is_tensor(z)
        assert x.shape == torch.Size([DATASET_SIZE, DATA_SIZE, DATA_SIZE])
        assert x.shape == y.shape
        assert z.shape == torch.Size([DATASET_SIZE, DATA_SIZE, 1])


class TensorDataset(Dataset):
    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        return index * torch.ones(DATA_SIZE, DATA_SIZE)


class TensorTwoTupleDataset(Dataset):
    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        return index * torch.ones(DATA_SIZE, DATA_SIZE), index * torch.ones(DATA_SIZE, 1)


class TensorThreeTupleDataset(Dataset):
    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        return index * torch.ones(DATA_SIZE, DATA_SIZE), \
               index * torch.ones(DATA_SIZE, DATA_SIZE), \
               index * torch.ones(DATA_SIZE, 1),
