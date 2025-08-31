import datasets
import torch
from torch.utils.data import IterableDataset
from typing import List, Union

class GiftEvalWrapperDataset(IterableDataset):
    def __init__(self, gift_eval_shard: IterableDataset):
        super().__init__()
        self.ds = gift_eval_shard
        

    def __iter__(self):
        dataset = iter(self.ds)
        while True:
            try:
                example = next(dataset)

                # in the first iteration we do not care about features; only about targets
                target = example["target"]
                
                
                if type(target[0]) == list:
                    for row in target:
                        yield self.__line_to_tensor(row)
                else:
                    yield self.__line_to_tensor(target)
            except StopIteration:
                return
            
    def __line_to_tensor(self, line: List) -> torch.Tensor:
        result = torch.unsqueeze(
            torch.tensor(line, dtype=torch.float32), 
            -1
        )

        return torch.nan_to_num(result, nan=0)

def load_gifteval_dataset(dataset_names: List) -> Union[
    datasets.DatasetDict, 
    datasets.Dataset, 
    datasets.IterableDatasetDict, datasets.IterableDataset
]:
    return datasets.load_dataset(
        "Salesforce/GiftEvalPretrain",
        split="train",
        streaming=True,
        data_files=[
            f'{name}/*.arrow' for name in dataset_names
        ]
    )

def load_gifteval_dataset_wrapper(dataset_names: List) -> GiftEvalWrapperDataset:
    return GiftEvalWrapperDataset(
        load_gifteval_dataset(
            dataset_names
        ) # type: ignore
    )
