import torch
import pytest
from collections import Counter
from typing import Dict

from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset
from preprocessing.common import TensorIterableDataset

class ConstantDataset(TensorIterableDataset):
    def __init__(self, value: float, count: int):
        self.value = value
        self.count = count

    def __iter__(self):
        for _ in range(self.count):
            yield torch.tensor([self.value], dtype=torch.float32)

def test_equal_probabilities_sampling():
    ds_a = ConstantDataset(1.0, 5)
    ds_b = ConstantDataset(2.0, 5)

    mixed_ds = ProbabilisticMixingDataset(
        datasets={"a": ds_a, "b": ds_b},
        probabilities=None  # should default to equal
    )

    outputs = list(mixed_ds)
    assert len(outputs) == 10

    values = [float(x.item()) for x in outputs]
    count = Counter(values)

    # Check all expected values are present
    assert count[1.0] == 5
    assert count[2.0] == 5

def test_weighted_probabilities_sampling():
    ds_a = ConstantDataset(1.0, 2)
    ds_b = ConstantDataset(2.0, 8)

    mixed_ds = ProbabilisticMixingDataset(
        datasets={"a": ds_a, "b": ds_b},
        probabilities={"a": 0.25, "b": 0.75}
    )

    outputs = list(mixed_ds)
    assert len(outputs) == 10

    values = [float(x.item()) for x in outputs]
    count = Counter(values)

    assert count[1.0] == 2
    assert count[2.0] == 8

def test_invalid_probability_sum():
    ds_a = ConstantDataset(1.0, 3)
    ds_b = ConstantDataset(2.0, 3)

    with pytest.raises(AssertionError, match="sum to 1"):
        ProbabilisticMixingDataset(
            datasets={"a": ds_a, "b": ds_b},
            probabilities={"a": 0.6, "b": 0.5}  # sums to 1.1
        )

def test_zero_probability():
    ds_a = ConstantDataset(1.0, 3)
    ds_b = ConstantDataset(2.0, 3)

    with pytest.raises(AssertionError, match="must be > 0"):
        ProbabilisticMixingDataset(
            datasets={"a": ds_a, "b": ds_b},
            probabilities={"a": 1.0, "b": 0.0}
        )

def test_reproducibility_with_seed():
    ds_a = ConstantDataset(1.0, 5)
    ds_b = ConstantDataset(2.0, 5)
    datasets: Dict[str, TensorIterableDataset] = {"a": ds_a, "b": ds_b}
    probabilities = {"a": 0.4, "b": 0.6}
    seed = 1337

    # First run
    ds1 = ProbabilisticMixingDataset(datasets, probabilities, seed)
    output1 = [float(x.item()) for x in ds1]

    # Second run
    ds2 = ProbabilisticMixingDataset(datasets, probabilities, seed)
    output2 = [float(x.item()) for x in ds2]

    assert output1 == output2, "Outputs should be identical for same seed"

