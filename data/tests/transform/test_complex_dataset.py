import torch
import pytest
from typing import List

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset
from preprocessing.transform.dataset_builder import Builder

class ConstantDataset(TensorIterableDataset):
    def __init__(self, values: List[torch.Tensor]):
        self.values = values

    def __iter__(self):
        return iter(self.values)

def test_complex_pipeline_reproducible_2d_with_builder():
    # Step 1: Create two datasets of 2D tensors
    dataset_1 = ConstantDataset([
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
    ])

    dataset_2 = ConstantDataset([
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
    ])

    # Step 2–4: Build dataset_1_final using Builder
    def transform_dataset_1(batch: torch.Tensor) -> torch.Tensor:
        linear_comb = 0.5 * batch[0] + 0.5 * batch[1]  # shape (2, 2)
        return linear_comb - linear_comb.mean()

    dataset_1_final = (
        Builder(dataset_1)
        .batch(batch_size=2)
        .map(transform_dataset_1)
        .build()
    )

    # Build dataset_2_final using Builder
    dataset_2_final = (
        Builder(dataset_2)
        .map(lambda ts: ts - ts.mean())
        .build()
    )

    # Step 5: Mix the datasets
    final_dataset = ProbabilisticMixingDataset(
        datasets={"ds1": dataset_1_final, "ds2": dataset_2_final},
        probabilities={"ds1": 0.5, "ds2": 0.5},
        seed=42,
    )

    result = list(final_dataset)

    # Expect 4 results (2 from each dataset)
    assert len(result) == 4

    for tensor in result:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert torch.isclose(tensor.mean(), torch.tensor(0.0), atol=1e-6)

    # Check reproducibility
    final_dataset_2 = ProbabilisticMixingDataset(
        datasets={"ds1": dataset_1_final, "ds2": dataset_2_final},
        probabilities={"ds1": 0.5, "ds2": 0.5},
        seed=42,
    )
    result2 = list(final_dataset_2)

    for t1, t2 in zip(result, result2):
        assert torch.allclose(t1, t2)
