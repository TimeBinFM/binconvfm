import itertools

import datasets
from binconvfm.utils.download.gift_eval_util import list_arrow_files
from binconvfm.utils.download.gift_eval_windowed_dataset import GiftEvalWindowedDataset
from binconvfm.utils.download.window_mixing_dataset import WindowMixingDataset
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule
import numpy as np


class FirstNDataset(IterableDataset):
    def __init__(self, dataset, n_rows):
        super().__init__()
        self.dataset = dataset
        self.n_rows = n_rows

    def __iter__(self):
        return iter(itertools.islice(self.dataset, self.n_rows))


class GiftEvalDataModule(LightningDataModule):
    def __init__(
        self,
        train_num_files: int = 1,
        val_num_files: int = 1,
        test_num_files: int = 1,
        predict_num_files: int = 1,
        n_rows: int = 1000,
        input_len: int = 32,
        random_seed: int = 1,
        batch_size: int = 64,
        output_len: int = 1,
        horizon: int = 10,
        step: int = 1,
        dataset_name: str = "Salesforce/GiftEvalPretrain",
        num_workers: int = 0,
        pre_batch_timeseries_count: int = 10,
        prefetch_depth: int = 1024,
    ):
        """
        DataModule for GiftEval dataset.
        Args:
            train_num_files (int): Number of files to use for training.
            val_num_files (int): Number of files to use for validation.
            test_num_files (int): Number of files to use for testing.
            predict_num_files (int): Number of files to use for prediction.
            n_rows (int): Number of rows to use from each file.
            input_len (int): Length of input sequences.
            random_seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for DataLoader.
            output_len (int): Length of output sequences.
            horizon (int): Prediction horizon.
            step (int): Step size for sliding window.
            dataset_name (str): Name of the dataset in Hugging Face Hub.
            num_workers (int): Number of workers for DataLoader.
            pre_batch_timeseries_count (int): Number of timeseries to batch together.
            prefetch_depth (int): Number of windows to prefetch
        """
        super().__init__()
        self.train_num_files = train_num_files
        self.val_num_files = val_num_files
        self.test_num_files = test_num_files
        self.predict_num_files = predict_num_files
        self.n_rows = n_rows
        self.input_len = input_len
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.output_len = output_len
        self.horizon = horizon
        self.step = step
        self.dataset_name = dataset_name
        self.file_names = list_arrow_files(dataset_name)
        self.num_workers = num_workers
        self.pre_batch_timeseries_count = pre_batch_timeseries_count
        self.prefetch_depth = prefetch_depth

    def setup(self, stage: str):
        np.random.seed(self.random_seed)
        if stage == "fit":
            train_file_names = np.random.choice(
                self.file_names, self.train_num_files, replace=False
            )
            val_file_names = np.random.choice(
                self.file_names, self.val_num_files, replace=False
            )
            self.train_ds = FirstNDataset(
                self._build_window_mixing_dataset(train_file_names),
                self.n_rows,
            )
            self.val_ds = FirstNDataset(
                self._build_window_mixing_dataset(val_file_names),
                self.n_rows,
            )
        elif stage == "test":
            test_file_names = np.random.choice(
                self.file_names, self.test_num_files, replace=False
            )
            self.test_ds = FirstNDataset(
                self._build_window_mixing_dataset(test_file_names),
                self.n_rows,
            )
        elif stage == "predict":
            predict_file_names = np.random.choice(
                self.file_names, self.predict_num_files, replace=False
            )
            self.predict_ds = FirstNDataset(
                self._build_window_mixing_dataset(predict_file_names),
                self.n_rows,
            )

    def _build_window_mixing_dataset(self, file_names: list[str]):
        windowed_datasets = [
            GiftEvalWindowedDataset(
                dataset_name=self.dataset_name,
                file_names=[file_name], 
                window_size=self.input_len, 
                prediction_depth=self.horizon, 
                step=self.step, 
                pre_batch_timeseries_count=self.pre_batch_timeseries_count
            )
            for file_name in file_names
        ]

        return WindowMixingDataset(
            windowed_datasets=windowed_datasets, 
            prediction_depth=self.horizon, 
            seed=self.random_seed, 
            batch_size=self.batch_size, 
            prefetch_depth=self.prefetch_depth
        )

        

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, shuffle=False, batch_size=None, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, shuffle=False, batch_size=None, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, shuffle=False, batch_size=None, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds, shuffle=False, batch_size=None, num_workers=self.num_workers
        )
