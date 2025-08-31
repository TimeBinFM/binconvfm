from torch.utils.data import IterableDataset
from typing import Iterator, Iterable, TypeVar

T = TypeVar('T')

class ConcatDataset(IterableDataset[T]):
    def __init__(self, datasets: Iterable[IterableDataset[T]]):
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator[T]:
        for dataset in self.datasets:
            for item in dataset:
                yield item
