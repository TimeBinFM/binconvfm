import random
import torch
from preprocessing.common import TensorIterableDataset
from typing import List
from torch.utils.data import IterableDataset, get_worker_info
from preprocessing.transform.dataset_builder import Builder
from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset

class WindowMixingDataset(IterableDataset):
    def __init__(self, windowed_datasets: List[TensorIterableDataset], window_size=32, prediction_depth=1, seed=42, batch_size=64, prefetch_depth=1024):
        super().__init__()

        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.seed = seed
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.windowed_datasets = windowed_datasets

        self.rng = random.Random(self.seed)

    def __build_worker_dataset(self, worker_id, num_workers):
        worker_windowed_datasets = self.windowed_datasets[worker_id::num_workers]

        dataset_dict = {
            f'stream_{i}': ds
            for i, ds in enumerate(worker_windowed_datasets)
        }

        mixed_dataset = ProbabilisticMixingDataset(dataset_dict, seed=self.seed)

        def shuffle_with_seed(lst):
            self.rng.shuffle(lst)
            return lst

        return (
            Builder(mixed_dataset)
                .batch(self.prefetch_depth) # prefetch prefetch_depth of items to randomly shuffle them
                .map(shuffle_with_seed)
                .flat()
                .batch(self.batch_size)
                .map(lambda batch: (torch.stack(batch)[:, :-self.prediction_depth], torch.stack(batch)[:, -self.prediction_depth:]))
                .build()
        )
        
    def __iter__(self):
        worker_info = get_worker_info()
        
        ds = self.__build_worker_dataset(
            worker_info.id, worker_info.num_workers
        ) if worker_info is not None else self.__build_worker_dataset(0, 1)
        
        for item in iter(ds):
            yield item
