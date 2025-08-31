import torch
import pytest

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.combining_dataset import CombiningDataset

# Dummy datasets that emit single-item tensors
class DummyDatasetA(TensorIterableDataset):
    def __iter__(self):
        for i in range(5):
            yield torch.tensor([i], dtype=torch.float32)

class DummyDatasetB(TensorIterableDataset):
    def __iter__(self):
        for i in range(5, 10):
            yield torch.tensor([i], dtype=torch.float32)

def test_combining_dataset_sum():
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()

    def add_fn(x, y):
        return x + y

    combined_ds = CombiningDataset([ds1, ds2], op=add_fn)

    expected = [torch.tensor([i + j], dtype=torch.float32)
                for i, j in zip(range(5), range(5, 10))]

    output = list(combined_ds)

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"
