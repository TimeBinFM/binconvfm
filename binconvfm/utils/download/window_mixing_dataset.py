import random
import torch
from preprocessing.common import TensorIterableDataset
from typing import List
from torch.utils.data import get_worker_info
from preprocessing.transform.dataset_builder import Builder
from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset

class WindowMixingDataset(TensorIterableDataset):
    def __init__(self, windowed_datasets: List[TensorIterableDataset], prediction_depth=8, seed=42, batch_size=64, prefetch_depth=1024):
        super().__init__()

        self.prediction_depth = prediction_depth
        self.seed = seed
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.windowed_datasets = windowed_datasets

        self.rng = random.Random(self.seed)

    def __get_worker_windowed_datasets(self, worker_id, num_workers):
        return self.windowed_datasets[worker_id::num_workers]

    def __build_probabilistic_mixing_dataset(self, worker_id, num_workers):
        worker_windowed_datasets = self.__get_worker_windowed_datasets(worker_id, num_workers)

        dataset_dict = {
            f'stream_{i}': ds
            for i, ds in enumerate(worker_windowed_datasets)
        }

        mixed_dataset = ProbabilisticMixingDataset(dataset_dict, seed=self.seed)
        return mixed_dataset

    def __build_worker_dataset(self, worker_id, num_workers):
        mixed_dataset = self.__build_probabilistic_mixing_dataset(worker_id, num_workers)

        def shufle_tensor_batch(batch):
            total_tensor = torch.cat(batch, dim=0)
            perm = torch.randperm(total_tensor.size(0))
            total_tensor_perm = total_tensor[perm]

            return torch.split(total_tensor_perm, self.batch_size)

        return (
            Builder(mixed_dataset)
                .batch(self.prefetch_depth) # prefetch prefetch_depth of items to randomly shuffle them
                .map(shufle_tensor_batch)
                .flat()
                .map(lambda batch: (batch[:, :-self.prediction_depth], batch[:, -self.prediction_depth:]))
                .build()
        )
        
    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        ds = self.__build_worker_dataset(worker_id, num_workers)
        
        for item in iter(ds):
            yield item
