import torch
import os
from torch.utils.data import Dataset
from typing import List

class SerializedTensorDataset(Dataset):
    def __init__(self, filepaths: List[str], lazy: bool = True):
        self.filepaths = sorted(filepaths)
        self.lazy = lazy
        self._index = []  # List of tuples: (file_idx, tensor_idx_within_file)

        if self.lazy:
            # Build index by scanning files without storing data
            for file_idx, path in enumerate(self.filepaths):
                data = torch.load(path, map_location="cpu")
                for i in range(len(data)):
                    self._index.append((file_idx, i))
                del data  # Explicitly free memory
        else:
            # Eager: load everything into memory
            self._tensors = []
            for path in self.filepaths:
                batch = torch.load(path, map_location="cpu")
                self._tensors.extend(batch)
            self._index = list(range(len(self._tensors)))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if self.lazy:
            file_idx, tensor_idx = self._index[idx]
            tensors = torch.load(self.filepaths[file_idx], map_location="cpu")
            return tensors[tensor_idx]
        else:
            return self._tensors[idx]
