from torch.utils.data import IterableDataset
from typing import Iterator, List, TypeVar

T = TypeVar('T')

class BatchingIterableDataset(IterableDataset[List[T]]):
    def __init__(self, dataset: IterableDataset[T], batch_size: int, 
                 include_last_batch: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.include_last_batch = include_last_batch

    def __iter__(self) -> Iterator[List[T]]:
        buffer: List[T] = []
        for item in iter(self.dataset):
            buffer.append(item)
            if len(buffer) == self.batch_size:
                yield buffer
                buffer = []
        
        if len(buffer) > 0 and self.include_last_batch:
            yield buffer
