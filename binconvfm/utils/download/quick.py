import datasets
import torch

from binconvfm.utils.download.gift_eval import list_arrow_files

def get_file_names_per_dataset(dataset_name: str):
    file_names = list_arrow_files(dataset_name)
    file_names_per_dataset = {}
    for file_name in file_names:
        dataset_name = file_name.split('/')[0]
        if dataset_name not in file_names_per_dataset:
            file_names_per_dataset[dataset_name] = []
        file_names_per_dataset[dataset_name].append(file_name)
    return file_names_per_dataset

def get_target_dataset(dataset_name: str, file_names_to_process: list[str]):
    ds = datasets.load_dataset(
        dataset_name,
        split='train',
        data_files=file_names_to_process,
        streaming=False,
    )
    ds.set_format(type='torch', columns=['target'])

    return ds

def dataset_to_window_tensors(dataset: datasets.Dataset, window_size: int, prediction_depth: int, step: int):
    tensor = torch.cat(list(dataset['target']), dim=0)
    return tensor.unfold(dimension=0, size=window_size+prediction_depth, step=step)

def get_windowed_target_dataset(base_dataset: datasets.Dataset | datasets.IterableDataset, batch_size, window_size, 
                       prediction_depth=1, step=1, pre_batch_ts_count=100):
    def item_to_window_tensor(target):
        target_tensor = torch.tensor(target)
        target_tensor_shape = target_tensor.shape

        unfold_dimension = len(target_tensor_shape) - 1 
        
        window_tensor = target_tensor.unfold(dimension=unfold_dimension, size=window_size+prediction_depth, step=step)
        
        window_tensor_shape = window_tensor.shape
        result_tensor = window_tensor if len(window_tensor_shape) == 2 else window_tensor.flatten(0, 1)

        return result_tensor

    def to_window_tensors(batch):
        tensors = [item_to_window_tensor(item) for item in batch['target']]
        total_tensor = torch.cat(tensors)
        return {
            'target': torch.split(total_tensor, batch_size)
        }

    return (
        base_dataset
            .select_columns(['target'])
            .batch(pre_batch_ts_count)
            .map(to_window_tensors)
    )
