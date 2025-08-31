from torch.utils.data import IterableDataset
from typing import Callable, Iterator, TypeVar

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')

class TransformingDataset(IterableDataset[T_out]):
    def __init__(
        self,
        ds: IterableDataset[T_in],
        op: Callable[[T_in], T_out],
    ):
        self.ds = ds
        self.op = op

    def __iter__(self) -> Iterator[T_out]:
        for item in iter(self.ds):
            yield self.op(item)
