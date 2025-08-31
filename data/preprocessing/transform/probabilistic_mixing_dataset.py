from torch.utils.data import IterableDataset
from typing import List, Dict, Iterator, Optional, TypeVar
import random

T = TypeVar('T')

class ProbabilisticMixingDataset(IterableDataset[T]):
    def __init__(
        self,
        datasets: Dict[str, IterableDataset[T]],
        probabilities: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.dataset_dict = datasets
        self.random = random.Random(seed)  # local RNG

        keys = list(datasets.keys())

        if probabilities is None:
            self.prob_dict = {k: 1.0 / len(keys) for k in keys}
        else:
            assert set(probabilities.keys()) == set(keys), "Keys of datasets and probabilities must match"
            assert all(p > 0 for p in probabilities.values()), "All probabilities must be > 0"
            total = sum(probabilities.values())
            assert abs(total - 1.0) < 1e-6, f"Probabilities must sum to 1. Got {total}"
            self.prob_dict = probabilities

    def __iter__(self) -> Iterator[T]:
        iterators: Dict[str, Iterator[T]] = {
            k: iter(ds) for k, ds in self.dataset_dict.items()
        }
        active_keys: List[str] = list(iterators.keys())

        while active_keys:
            active_probs = [self.prob_dict[k] for k in active_keys]
            total = sum(active_probs)
            normalized_probs = [p / total for p in active_probs]

            chosen_key = self.random.choices(active_keys, weights=normalized_probs, k=1)[0]

            try:
                yield next(iterators[chosen_key])
            except StopIteration:
                active_keys.remove(chosen_key)
