import torch
import os
from ..common import TensorIterableDataset

def serialize_tensor_stream(
    dataset: TensorIterableDataset,
    output_dir: str,
    max_tensors_per_file: int = 1000,
    filename_prefix: str = "shard",
):
    os.makedirs(output_dir, exist_ok=True)
    file_index = 0
    buffer = []

    for tensor in dataset:
        buffer.append(tensor)

        if len(buffer) == max_tensors_per_file:
            file_path = os.path.join(output_dir, f"{filename_prefix}_{file_index:05d}.pt")
            torch.save(buffer, file_path)
            buffer.clear()
            file_index += 1

    # Save remaining tensors
    if buffer:
        file_path = os.path.join(output_dir, f"{filename_prefix}_{file_index:05d}.pt")
        torch.save(buffer, file_path)
