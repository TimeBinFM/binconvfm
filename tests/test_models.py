import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from binconvfm import models as _models
from inspect import getmembers, isclass
from binconvfm.models.base import BaseForecaster, BaseLightningModule
from binconvfm.models.lstm import LSTMForecaster

# Collect all model classes from __init__.py
model_classes = [f[1] for f in getmembers(_models, isclass)]


def create_model(ModelClass, input_len, n_samples, **kwargs):
    """Create a model instance with appropriate parameters based on model type."""
    if ModelClass.__name__ == "BinConvForecaster":
        return ModelClass(
            context_length=input_len,
            num_filters_2d=input_len,
            num_filters_1d=input_len,
            num_bins=1024,
            num_blocks=1,
            n_samples=n_samples,
            **kwargs,
        )
    else:
        return ModelClass(n_samples=n_samples, **kwargs)


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
        self.n_samples = 2
        self.input_len = 20
        train_ds = DummyDataset(input_len=self.input_len, output_len=1)
        test_ds = DummyDataset(input_len=self.input_len, output_len=self.horizon)

        self.train_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=False
        )
        self.test_dataloader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False
        )
        self.pred_dataloader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False
        )

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_init(self, ModelClass):
        self.model = ModelClass(n_samples=self.n_samples)
        self.model._create_model()
        assert isinstance(
            self.model, BaseForecaster
        ), f"{self.model} is not a BaseForecaster"
        assert isinstance(
            self.model.model, BaseLightningModule
        ), f"{self.model.model} is not a BaseLightningModule"
        assert isinstance(
            self.model.model.model, nn.Module
        ), f"{self.model.model.model} is not a nn.Module"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_fit(self, ModelClass):
        self.model = create_model(
            ModelClass, self.input_len, n_samples=self.n_samples, num_epochs=1
        )
        self.model.fit(self.train_dataloader, self.val_dataloader)
        assert True, "Model should fit without errors"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_evaluate(self, ModelClass):
        self.model = create_model(ModelClass, self.input_len, n_samples=self.n_samples)
        metrics = self.model.evaluate(self.test_dataloader)
        assert isinstance(metrics, dict), "Result should be a dictionary"
        assert metrics["mase"] > 0, "MASE should be positive"
        assert metrics["crps"] > 0, "CRPS should be positive"
        assert metrics["nmae"] > 0, "NMAE should be positive"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_predict(self, ModelClass):
        self.model = create_model(ModelClass, self.input_len, n_samples=self.n_samples)
        pred = self.model.predict(self.pred_dataloader, self.horizon)
        assert isinstance(pred, list), "Prediction should be a list"
        assert len(pred) == len(self.pred_dataloader), "Prediction length mismatch"
        different_samples = False
        for i in range(len(pred) - 1):
            p = pred[i]
            assert isinstance(p, torch.Tensor), "Each prediction should be a Tensor"
            assert p.shape == (
                self.batch_size,
                self.n_samples,
                self.horizon,
                1,
            ), "Each prediction should have shape (batch_size, n_samples, horizon, dim)"
            different_samples = different_samples or not torch.allclose(
                p[:, 0, :, :], p[:, 1, :, :]
            )
        assert different_samples, "Samples should be different"

    def test_transform_factory(self):
        self.model = LSTMForecaster(
            n_samples=self.n_samples, transform=["StandardScaler"]
        )
        self.model._create_model()
        # Transform should be a pipeline with StandardScaler
        assert len(self.model.model.transform.steps) == 1
        assert self.model.model.transform.steps[0][0] == "StandardScaler"

        # Create a sample batch to test transform
        sample_data = torch.randn(self.batch_size, 20, 1)  # (batch, seq_len, features)
        transformed, params = self.model.model.transform.fit_transform(sample_data)

        # Check that parameters have correct shapes (per-sample scaling)
        assert params[0]["mean"].shape == (self.batch_size, 1, 1)
        assert params[0]["std"].shape == (self.batch_size, 1, 1)

        self.model.fit(self.train_dataloader, self.val_dataloader)
        self.model.evaluate(self.test_dataloader)
        assert True, "Model should evaluate without errors"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_save_and_load_checkpoint(self, tmp_path, ModelClass):
        self.model = create_model(
            ModelClass, self.input_len, n_samples=self.n_samples, num_epochs=1
        )
        self.model.fit(self.train_dataloader, self.val_dataloader)

        ckpt_path = tmp_path / "model.ckpt"
        self.model.save_checkpoint(str(ckpt_path))

        # Load the model from checkpoint
        new_model = ModelClass(n_samples=self.n_samples)
        new_model._create_model()
        new_model.load_checkpoint(str(ckpt_path))

        # Compare model parameters to ensure they are the same
        for p1, p2 in zip(self.model.model.parameters(), new_model.model.parameters()):
            assert torch.allclose(
                p1, p2, atol=1e-6
            ), "Model weights do not match after loading checkpoint"
