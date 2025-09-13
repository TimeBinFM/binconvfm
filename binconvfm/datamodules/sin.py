from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import numpy as np


class SinDataset(Dataset):
    def __init__(self, input_len, output_len):
        x_space = np.linspace(0, 100, 1000)
        seq = np.sin(x_space) + np.random.randn(1000) * 0.1
        self.input_len = input_len
        self.output_len = output_len
        self.seq = torch.tensor(seq, dtype=torch.float32)
        self.seq = self.seq.unsqueeze(-1)
        self.length = len(seq) - input_len - output_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_seq = self.seq[idx : idx + self.input_len]
        target_seq = self.seq[
            idx + self.input_len : idx + self.input_len + self.output_len
        ]
        return input_seq, target_seq


class SinDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        horizon: int,
        input_len: int,
        output_len: int,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.horizon = horizon

    def setup(self, stage: str):

        if stage == "fit":
            self.train_ds = SinDataset(
                input_len=self.input_len, output_len=self.output_len
            )
            self.val_ds = SinDataset(
                input_len=self.input_len, output_len=self.output_len
            )
        elif stage == "test":
            self.test_ds = SinDataset(input_len=self.input_len, output_len=self.horizon)
        elif stage == "predict":
            self.pred_ds = SinDataset(input_len=self.input_len, output_len=self.horizon)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=self.batch_size)
