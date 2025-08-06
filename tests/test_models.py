import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from binconvfm import models as _models
from inspect import getmembers, isclass
from binconvfm.models.base import BaseForecaster, BaseLightningModule
import torch.nn as nn

# Collect all model classes from __init__.py
model_classes = [f[1] for f in getmembers(_models, isclass)]


class DummyDataset(Dataset):
    def __init__(self, input_len=20, output_len=5):
        self.input_len = input_len
        self.output_len = output_len
        self.seq = torch.randn(1000, dtype=torch.float32)
        self.length = len(self.seq) - input_len - output_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_seq = self.seq[idx : idx + self.input_len]
        target_seq = self.seq[
            idx + self.input_len : idx + self.input_len + self.output_len
        ]
        return input_seq, target_seq


class TestOnDummyDataset:
    def setup_class(self):
        self.horizon = 5
        ds = DummyDataset(output_len=self.horizon)
        self.train_dataloader = DataLoader(ds, batch_size=32, shuffle=True)
        self.val_dataloader = DataLoader(ds, batch_size=32, shuffle=False)
        self.test_dataloader = DataLoader(ds, batch_size=32, shuffle=False)
        self.pred_dataloader = DataLoader(ds, batch_size=32, shuffle=False)

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_init(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon)
        self.model._create_model()
        assert isinstance(
            self.model, BaseForecaster
        ), f"{self.model} is not a BaseForecaster"
        assert isinstance(
            self.model.model, BaseLightningModule
        ), f"{self.model.model} is not a BaseForecasterModule"
        assert isinstance(
            self.model.model.model, nn.Module
        ), f"{self.model.model.model} is not a nn.Module"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_fit(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon)
        self.model.fit(self.train_dataloader, self.val_dataloader)

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_evaluate(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon)
        result = self.model.evaluate(self.test_dataloader)
        assert result is not None

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_predict(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon)
        preds = self.model.predict(self.pred_dataloader)
        assert preds is not None
