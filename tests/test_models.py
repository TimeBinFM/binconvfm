import pytest
import torch
import torch.nn as nn
from binconvfm import models as _models
from binconvfm.datamodules import DummyDatamodule
from inspect import getmembers, isclass
from binconvfm.models.base import BaseForecaster, BaseLightningModule
from binconvfm.models.lstm import LSTMForecaster

# Collect all model classes from __init__.py
model_classes = [f[1] for f in getmembers(_models, isclass)]


class TestOnDummyDataset:

    def setup_class(self):
        self.input_len = 20
        self.output_len = 1
        self.horizon = 5
        self.batch_size = 32
        self.n_samples = 1000
        self.datamodule = DummyDatamodule(
            batch_size=self.batch_size,
            horizon=self.horizon,
            n_samples=self.n_samples,
            input_len=self.input_len,
            output_len=self.output_len,
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
        self.model = ModelClass(n_samples=self.n_samples)
        self.model.fit(self.datamodule)
        assert True, "Model should fit without errors"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_evaluate(self, ModelClass):
        self.model = ModelClass(n_samples=self.n_samples)
        metrics = self.model.evaluate(self.datamodule)
        assert isinstance(metrics, dict), "Result should be a dictionary"
        assert metrics["mase"] > 0, "MASE should be positive"
        assert metrics["crps"] > 0, "CRPS should be positive"
        assert metrics["nmae"] > 0, "NMAE should be positive"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_predict(self, ModelClass):
        self.model = ModelClass(n_samples=self.n_samples)
        pred = self.model.predict(self.datamodule, self.horizon)
        assert isinstance(pred, list), "Prediction should be a list"
        assert (
            len(pred) == len(self.datamodule.pred_ds) // self.batch_size + 1
        ), "Prediction length mismatch"
        for p in pred[:-1]:
            assert isinstance(p, torch.Tensor), "Each prediction should be a Tensor"
            assert p.shape == (
                self.batch_size,
                self.model.n_samples,
                self.horizon,
                1,
            ), "Each prediction should have shape (batch_size, n_samples, horizon, dim)"

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

        self.model.fit(self.datamodule)
        self.model.evaluate(self.datamodule)
        assert True, "Model should evaluate without errors"

    @pytest.mark.parametrize("ModelClass", model_classes)
    def test_save_and_load_checkpoint(self, tmp_path, ModelClass):
        self.model = ModelClass(n_samples=self.n_samples)
        self.model.fit(self.datamodule)

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
