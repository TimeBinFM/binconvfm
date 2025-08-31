import torch
import pytest
from typing import List
from torch.utils.data import IterableDataset

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.unbatching_dataset import UnbatchingIterableDataset

# Dummy dataset that yields two batches
class DummyBatchDataset(IterableDataset[List[torch.Tensor]]):
    def __iter__(self):
        yield [torch.tensor([1, 2]), torch.tensor([3, 4])]  # shape: (2, 2)
        yield [torch.tensor([5, 6]), torch.tensor([7, 8])]  # shape: (2, 2)

def test_unbatching_iterable_dataset_dim0():
    ds = DummyBatchDataset()
    unbatch_ds = UnbatchingIterableDataset(ds)

    output = list(unbatch_ds)

    expected = [
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.tensor([5, 6]),
        torch.tensor([7, 8]),
    ]

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"
