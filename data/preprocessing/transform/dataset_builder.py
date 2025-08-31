from typing import Callable, TypeVar, Generic
from torch.utils.data import IterableDataset

from ..common import TensorIterableDataset
from .transforming_dataset import TransformingDataset
from .batching_dataset import BatchingIterableDataset
from .unbatching_dataset import UnbatchingIterableDataset
from .sliding_window_dataset import SlidingWindowIterableDataset

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')


class Builder(Generic[T_in]):
    def __init__(self, dataset: IterableDataset[T_in]):
        self.dataset = dataset

    def map(self, op: Callable[[T_in], T_out]) -> "Builder":
        return Builder(TransformingDataset(self.dataset, op=op))

    def batch(self, batch_size: int, include_last_batch: bool = True) -> "Builder":
        return Builder(BatchingIterableDataset(self.dataset, batch_size, include_last_batch))
    
    def sliding_window(self, window_size: int, step: int = 1) -> "Builder":
        if not isinstance(self.dataset, TensorIterableDataset):
            raise ValueError("Dataset must be a TensorIterableDataset")

        return Builder(SlidingWindowIterableDataset(self.dataset, window_size, step))

    def flat(self) -> "Builder":
        return Builder(UnbatchingIterableDataset(self.dataset))  # pyright: ignore[reportArgumentType]

    def build(self) -> IterableDataset[T_in]:
        return self.dataset
