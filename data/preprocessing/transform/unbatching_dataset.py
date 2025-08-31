from torch.utils.data import IterableDataset
from typing import List, Iterator, TypeVar

T = TypeVar('T')

class UnbatchingIterableDataset(IterableDataset[List[T]]):
    def __init__(self, dataset: IterableDataset[List[T]]):
        self.dataset = dataset

    def __iter__(self) -> Iterator[T]:
        for batch in iter(self.dataset):
            for item in batch:
                yield item
