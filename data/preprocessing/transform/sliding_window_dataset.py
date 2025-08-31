from torch.utils.data import IterableDataset
from ..common import TensorIterableDataset
from torch import Tensor
from typing import Iterator

class SlidingWindowIterableDataset(IterableDataset):
    def __init__(self, dataset: TensorIterableDataset, window_size: int, step: int = 1):
        self.dataset = dataset
        self.window_size = window_size
        self.step = step

    def __iter__(self) -> Iterator[Tensor]:
        iterator = iter(self.dataset)

        for tensor in iterator:
            for i in range(len(tensor)):
                left = i
                right = i + self.window_size

                if right > len(tensor):
                    break

                yield tensor[left:right]
