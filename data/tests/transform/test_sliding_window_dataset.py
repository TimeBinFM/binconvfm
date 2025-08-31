import torch
from torch.utils.data import IterableDataset
import pytest

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.sliding_window_dataset import SlidingWindowIterableDataset

# Dummy dataset that yields fixed tensors
class DummyTensorDataset(TensorIterableDataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __iter__(self):
        for tensor in self.tensors:
            yield tensor


@pytest.mark.parametrize("tensors, window_size, step, expected", [
    (
        [torch.tensor([1, 2, 3, 4, 5])],
        3,
        1,
        [torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]), torch.tensor([3, 4, 5])]
    ),
    (
        [torch.tensor([1, 2, 3, 4, 5])],
        2,
        1,
        [torch.tensor([1, 2]), torch.tensor([2, 3]), torch.tensor([3, 4]), torch.tensor([4, 5])]
    ),
    (
        [torch.tensor([1, 2, 3])],
        4,
        1,
        []  # window too large
    ),
    (
        [torch.tensor([1, 2, 3, 4])],
        2,
        2,
        [torch.tensor([1, 2]), torch.tensor([2, 3]), torch.tensor([3, 4])]
    ),
])
def test_sliding_window_iterable_dataset(tensors, window_size, step, expected):
    dataset = DummyTensorDataset(tensors)
    sliding = SlidingWindowIterableDataset(dataset, window_size, step)
    result = list(sliding)

    assert len(result) == len(expected)
    for res, exp in zip(result, expected):
        assert torch.equal(res, exp)
