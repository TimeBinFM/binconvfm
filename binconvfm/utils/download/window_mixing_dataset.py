import random
import torch
from preprocessing.common import TensorIterableDataset
from preprocessing.transform.dataset_builder import Builder
from binconvfm.utils.download.gift_eval_windowed_dataset import GiftEvalWindowedDataset

class WindowMixingDataset(TensorIterableDataset):
    def __init__(self, windowed_dataset: GiftEvalWindowedDataset, prediction_depth=8, seed=42, batch_size=64, prefetch_depth=1024):
        super().__init__()

        self.prediction_depth = prediction_depth
        self.seed = seed
        self.batch_size = batch_size
        self.prefetch_depth = prefetch_depth
        self.windowed_dataset = windowed_dataset

        self.rng = random.Random(self.seed)
        
    def __iter__(self):
        def shufle_tensor_batch(batch):
            total_tensor = torch.cat(batch, dim=0)
            perm = torch.randperm(total_tensor.size(0))
            total_tensor_perm = total_tensor[perm]

            return torch.split(total_tensor_perm, self.batch_size)

        ds = (
            Builder(self.windowed_dataset)
                .batch(self.prefetch_depth) # prefetch prefetch_depth of items to randomly shuffle them
                .map(shufle_tensor_batch)
                .flat()
                .map(lambda batch: (batch[:, :-self.prediction_depth], batch[:, -self.prediction_depth:]))
                .build()
        )
        
        for item in iter(ds):
            yield item
