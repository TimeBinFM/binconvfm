import re
import torch
from huggingface_hub import list_repo_files
import datasets
from torch.utils.data import IterableDataset, get_worker_info
from data.preprocessing.transform.dataset_builder import Builder
from data.preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset
from data.preprocessing.downloader.gift_eval import GiftEvalWrapperDataset


def list_arrow_files(dataset_name):
    # List all files in the Hugging Face dataset repo
    files = list_repo_files(dataset_name, repo_type="dataset")
    
    # Filter files matching the pattern
    pattern = re.compile(r".*data-\d+-of-\d+\.arrow$")
    filtered_files = [f for f in files if pattern.match(f)]
    
    return filtered_files

class PostProcessingDataset(IterableDataset):
    def __init__(self, file_names, window_size=32, prediction_depth=1, seed=42, batch_size=64, num_workers=1, split='train', 
                 dataset_name="Salesforce/GiftEvalPretrain"):
        super().__init__()

        self.file_names = file_names
        self.window_size = window_size
        self.prediction_depth = prediction_depth
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.split = split
        self.dataset_name = dataset_name

        self.worker_datasets = [self.__build_worker_dataset(worker_id) for worker_id in range(num_workers)]

    def __build_worker_dataset(self, worker_id):
        worker_file_names = self.__get_worker_file_names(worker_id)
        worker_file_names_left, worker_file_names_right = self.__split_file_names(worker_file_names)

        file_name_dict = {
            'left': worker_file_names_left,
            'right': worker_file_names_right
        }

        dataset_dict = {
            name: (
                Builder(self.__load_dataset(file_names))
                .sliding_window(self.window_size + self.prediction_depth)
                .map(lambda t: (t[:self.window_size], t[self.window_size:]))
                .build()
            )
            for name, file_names in file_name_dict.items()
        }

        mixed_dataset = ProbabilisticMixingDataset(dataset_dict, seed=self.seed)

        def collate_list_of_tuples(data):
            features, targets = zip(*data)
            features = torch.stack(features)
            targets = torch.stack(targets)
            return features, targets

        return (
            Builder(mixed_dataset)
                .batch(self.batch_size)
                .map(collate_list_of_tuples)
                .build()
        )
        

    def __load_dataset(self, file_names):
        return GiftEvalWrapperDataset(
            datasets.load_dataset(
                self.dataset_name,
                split=self.split,
                data_files=file_names
            )
        )

    def __split_file_names(self, file_names):
        half_length = len(file_names) // 2
        return file_names[:half_length], file_names[half_length:]

    def __get_worker_file_names(self, worker_id):
        if self.num_workers == 1:
            return self.file_names
        
        per_worker = int(len(self.file_names) / self.num_workers)
        
        start = worker_id * per_worker
        # Last worker might take the remainder
        end = (worker_id + 1) * per_worker if worker_id != self.num_workers - 1 else len(self.file_names)

        return self.file_names[start:end]

        
    def __iter__(self):
        ds = self.worker_datasets[self.__get_worker_id()]
        
        for item in iter(ds):
            yield item

    def __get_worker_id(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return 0

        assert self.num_workers == self.num_workers, 'dataset numm_workers must match the passed num_workers'
        
        return worker_info.id
