from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch


class DummyDataset(Dataset):
    def __init__(self, input_len, output_len):
        self.input_len = input_len
        self.output_len = output_len
        torch.manual_seed(0)  # For reproducibility
        self.seq = torch.randn((1000, 1), dtype=torch.float32)
        self.length = len(self.seq) - self.input_len - self.output_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_seq = self.seq[idx : idx + self.input_len]
        target_seq = self.seq[
            idx + self.input_len : idx + self.input_len + self.output_len
        ]
        return input_seq, target_seq


class DummyDataModule(LightningDataModule):
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
            self.train_ds = DummyDataset(
                input_len=self.input_len, output_len=self.output_len
            )
            self.val_ds = DummyDataset(
                input_len=self.input_len, output_len=self.output_len
            )
        elif stage == "test":
            self.test_ds = DummyDataset(
                input_len=self.input_len, output_len=self.horizon
            )
        elif stage == "predict":
            self.pred_ds = DummyDataset(
                input_len=self.input_len, output_len=self.horizon
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=self.batch_size)
