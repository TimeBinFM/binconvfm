from torch.utils.data import IterableDataset
from typing import Callable, Iterator, Iterable, TypeVar    

T = TypeVar('T')

class CombiningDataset(IterableDataset[T]):
    def __init__(
        self,
        datasets: Iterable[IterableDataset],
        op: Callable[..., T],
    ):
        self.datasets = list(datasets)
        self.op = op

    def __iter__(self) -> Iterator[T]:
        iterators = [iter(ds) for ds in self.datasets]
        for items in zip(*iterators):
            yield self.op(*items)
