import datasets
import torch
from binconvfm.utils.download.gift_eval_pretrain_file_names import gift_eval_pretrain_file_names
from huggingface_hub import list_repo_files
import re

def list_arrow_files(dataset_name):
    if dataset_name == "Salesforce/GiftEvalPretrain":
        return gift_eval_pretrain_file_names

    # List all files in the Hugging Face dataset repo
    files = list_repo_files(dataset_name, repo_type="dataset")
    
    # Filter files matching the pattern
    pattern = re.compile(r".*data-\d+-of-\d+\.arrow$")
    filtered_files = [f for f in files if pattern.match(f)]
    
    return filtered_files

def get_file_names_per_dataset(dataset_name: str):
    file_names = list_arrow_files(dataset_name)
    file_names_per_dataset = {}
    for file_name in file_names:
        dataset_name = file_name.split('/')[0]
        if dataset_name not in file_names_per_dataset:
            file_names_per_dataset[dataset_name] = []
        file_names_per_dataset[dataset_name].append(file_name)
    return file_names_per_dataset

def dataset_to_window_tensors(dataset: datasets.Dataset, window_size: int, prediction_depth: int, step: int):
    tensor = torch.cat(list(dataset['target']), dim=0)
    return tensor.unfold(dimension=0, size=window_size+prediction_depth, step=step)
