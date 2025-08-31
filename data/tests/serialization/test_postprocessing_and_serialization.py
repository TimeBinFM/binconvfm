import tempfile
import shutil
import torch
from typing import List, Iterator

from pathlib import Path

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.dataset_builder import Builder
from preprocessing.serialization.serialize import serialize_tensor_stream
from preprocessing.serialization.deserialize import SerializedTensorDataset


class ConstantDataset(TensorIterableDataset):
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensors = tensors

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.tensors)

def test_serialization_and_pipeline_reproducibility():
    # Step 1: Create dataset_1 of 2D tensors
    dataset_1 = ConstantDataset([
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
    ])

    # Step 2: Transform dataset_1 using Builder
    dataset_1_final = (
        Builder(dataset_1)
        .map(lambda t: t - t.mean())
        .build()
    )

    # Step 3: Serialize the transformed dataset to disk
    tmp_dir = tempfile.mkdtemp()
    try:
        serialize_tensor_stream(dataset_1_final, tmp_dir, max_tensors_per_file=1)

        # Step 4: Load it back using SerializedTensorDataset
        shard_files = sorted(Path(tmp_dir).glob("*.pt"))
        result = SerializedTensorDataset([str(p) for p in shard_files], lazy=True)

        assert len(result) == 4

        for tensor in result:
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (2, 2)
            assert torch.isclose(tensor.mean(), torch.tensor(0.0), atol=1e-6)

    finally:
        shutil.rmtree(tmp_dir)
