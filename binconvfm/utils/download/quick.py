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
