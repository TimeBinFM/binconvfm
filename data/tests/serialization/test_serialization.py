import tempfile
import shutil
import torch
from torch.utils.data import IterableDataset
from pathlib import Path
from typing import List

from preprocessing.serialization.serialize import serialize_tensor_stream
from preprocessing.serialization.deserialize import SerializedTensorDataset

class DummyTensorStream(IterableDataset):
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensors = tensors

    def __iter__(self):
        return iter(self.tensors)

def test_serialize_tensor_stream_and_dataset():
    # Create a dummy stream of 10 small tensors
    tensors = [torch.tensor([i], dtype=torch.float32) for i in range(10)]

    stream = DummyTensorStream(tensors)

    tmp_dir = tempfile.mkdtemp()

    try:
        # Serialize with 4 tensors per file -> should create 3 files
        serialize_tensor_stream(stream, tmp_dir, max_tensors_per_file=4)

        files = sorted(Path(tmp_dir).glob("*.pt"))
        assert len(files) == 3

        # Check the file contents directly
        all_loaded = []
        for f in files:
            batch = torch.load(f)
            all_loaded.extend(batch)

        for t1, t2 in zip(all_loaded, tensors):
            assert torch.allclose(t1, t2)

        # Check lazy dataset
        lazy_ds = SerializedTensorDataset([str(f) for f in files], lazy=True)
        assert len(lazy_ds) == len(tensors)
        for i, ref in enumerate(tensors):
            assert torch.allclose(lazy_ds[i], ref)

        # Check eager dataset
        eager_ds = SerializedTensorDataset([str(f) for f in files], lazy=False)
        assert len(eager_ds) == len(tensors)
        for i, ref in enumerate(tensors):
            assert torch.allclose(eager_ds[i], ref)
    finally:
        shutil.rmtree(tmp_dir)
