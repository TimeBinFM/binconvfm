import torch
import pytest

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.batching_dataset import BatchingIterableDataset

# Dummy dataset that yields 7 scalar tensors: [0], [1], ..., [6]
class DummyDataset(TensorIterableDataset):
    def __iter__(self):
        for i in range(7):
            yield torch.tensor([i], dtype=torch.float32)

@pytest.mark.parametrize("batch_size, include_last_batch, expected_batches", [
    (3, True, [
        [torch.tensor([0.]), torch.tensor([1.]), torch.tensor([2.])],
        [torch.tensor([3.]), torch.tensor([4.]), torch.tensor([5.])],
        [torch.tensor([6.])]
    ]),
    (3, False, [
        [torch.tensor([0.]), torch.tensor([1.]), torch.tensor([2.])],
        [torch.tensor([3.]), torch.tensor([4.]), torch.tensor([5.])],
    ]),
    (1, True, [[torch.tensor([i], dtype=torch.float32)] for i in range(7)]),
])
def test_batching_iterable_dataset(batch_size, include_last_batch, expected_batches):
    ds = DummyDataset()
    batching_ds = BatchingIterableDataset(ds, batch_size, include_last_batch)
    
    output_batches = list(batching_ds)

    assert len(output_batches) == len(expected_batches)
    for output_batch, exp_batch in zip(output_batches, expected_batches):
        for out, exp in zip(output_batch, exp_batch):
            assert torch.equal(out, exp), f"Expected {exp}, got {out}"
