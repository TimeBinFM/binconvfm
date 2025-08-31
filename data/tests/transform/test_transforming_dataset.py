import torch
import pytest
from preprocessing.common import TensorIterableDataset
from preprocessing.transform.transforming_dataset import TransformingDataset

class DummyDataset(TensorIterableDataset):
    def __iter__(self):
        for i in range(5):
            yield torch.tensor([i], dtype=torch.float32)

def test_transforming_dataset_applies_function():
    base_ds = DummyDataset()
    op = lambda x: x * 2

    ds = TransformingDataset(base_ds, op)

    expected = [torch.tensor([0.]), torch.tensor([2.]), torch.tensor([4.]),
                torch.tensor([6.]), torch.tensor([8.])]
    output = list(ds)

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"
