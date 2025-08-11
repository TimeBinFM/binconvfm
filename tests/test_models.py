import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from binconvfm import models as _models
from inspect import getmembers, isclass
from binconvfm.models.base import BaseForecaster, BaseLightningModule, BaseTorchModule
from binconvfm.models.lstm import LSTMForecaster

# Collect all model classes from __init__.py
model_classes = [f[1] for f in getmembers(_models, isclass)]


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


class TestOnDummyDataset:
    def setup_class(self):
        self.horizon = 5
        self.batch_size = 32
        self.n_samples = 1000
        ds = DummyDataset(input_len=20, output_len=self.horizon)
        self.train_dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        self.pred_dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_init(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon, n_samples=self.n_samples)
        self.model._create_model()
        assert isinstance(
            self.model, BaseForecaster
        ), f"{self.model} is not a BaseForecaster"
        assert isinstance(
            self.model.model, BaseLightningModule
        ), f"{self.model.model} is not a BaseLightningModule"
        assert isinstance(
            self.model.model.model, BaseTorchModule
        ), f"{self.model.model.model} is not a BaseTorchModule"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_fit(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon, n_samples=self.n_samples)
        self.model.fit(self.train_dataloader, self.val_dataloader)
        assert True, "Model should fit without errors"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_evaluate(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon, n_samples=self.n_samples)
        metrics = self.model.evaluate(self.test_dataloader)
        assert isinstance(metrics, dict), "Result should be a dictionary"
        assert metrics["mase"] == pytest.approx(0.77, abs=1e-2)
        assert metrics["crps"] == pytest.approx(0.41, abs=1e-2)

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_predict(self, ModelClass):
        self.model = ModelClass(horizon=self.horizon)
        pred = self.model.predict(self.pred_dataloader)
        assert isinstance(pred, list), "Prediction should be a list"
        assert len(pred) == len(self.pred_dataloader), "Prediction length mismatch"
        for p in pred[:-1]:
            assert isinstance(p, torch.Tensor), "Each prediction should be a Tensor"
            assert p.shape == (
                self.batch_size,
                self.model.n_samples,
                self.horizon,
                1,
            ), "Each prediction should have shape (batch_size, n_samples, horizon, dim)"

    def test_standard_scaler(self):
        self.model = LSTMForecaster(horizon=self.horizon, transform="standard_scaler")
        self.model.fit(self.train_dataloader, self.val_dataloader)
        self.model.evaluate(self.test_dataloader)
        assert True, "Model should evaluate without errors"
