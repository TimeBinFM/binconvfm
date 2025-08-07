from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import LightningModule
import torch.nn as nn
from binconvfm.utils.metrics import mase, crps
from abc import abstractmethod


class BaseForecaster:
    def __init__(
        self,
        horizon: int = 1,
        n_samples: int = 1000,
        quantiles: list[float] = [(i + 1) / 10 for i in range(9)],
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        accelerator: str = "cpu",
        enable_progress_bar: bool = True,
        logging: bool = False,
        log_every_n_steps: int = 10,
    ):
        self.horizon = horizon
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logger = False
        if logging:
            self.logger = CSVLogger(save_dir="logs")
        self.log_every_n_steps = log_every_n_steps
        self.trainer = None
        self.model = None

    @abstractmethod
    def _create_model(self):
        pass

    def set_horizon(self, horizon):
        self.horizon = horizon
        if self.model is not None:
            self.model.horizon = horizon

    def set_n_samples(self, n_samples):
        self.n_samples = n_samples
        if self.n_samples is not None:
            self.model.n_samples = n_samples

    def _create_trainer(self):
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=self.log_every_n_steps,
            logger=self.logger,
            accelerator=self.accelerator,
        )

    def fit(self, train_dataloader, val_dataloader=None):
        self._create_trainer()
        self._create_model()
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate(self, test_dataloader):
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        metrics = self.trainer.test(self.model, test_dataloader)
        return metrics[0] # Assuming single test dataloader and single result

    def predict(self, pred_dataloader):
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        return self.trainer.predict(self.model, pred_dataloader)


class BaseLightningModule(LightningModule):
    def __init__(self, horizon, n_samples, quantiles, lr):
        """
        PyTorch Lightning module for LSTM forecasting.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.horizon = horizon
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        pred_seq = self.model(
            input_seq, self.horizon, self.n_samples
        )  # (batch, n_samples, output_len)
        metrics = {
            "mase": mase(pred_seq, target_seq),
            "crps": crps(pred_seq, target_seq, self.quantiles),
        }
        self.log_dict(metrics, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        input_seq, _ = batch
        return self.model(input_seq, self.horizon, self.n_samples)
    
    @abstractmethod
    def loss(self, batch, batch_idx):
        """
        Compute the loss for a batch.

        Args:
            batch (Tensor): Tuple of (input_seq, target_seq).
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss value.
        """
        raise NotImplementedError("Subclasses must implement this method")


class BaseTorchModule(nn.Module):
    def forward(self, x, horizon, n_samples, y=None):
        """
        Vectorized forward pass for the LSTM model with random sampling.

        Args:
            x (Tensor): Input sequence of shape (batch, input_len, dim).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.
            y (Tensor, optional): Teacher-forced target sequence (batch, output_len, dim).

        Returns:
            Tensor: Forecast samples of shape (batch, n_samples, output_len, dim).
        """
        assert x.dim() == 3, "Input must be a 3D tensor (batch, input_len, dim)"
        assert (
            y is None or y.dim() == 3
        ), "Target must be a 3D tensor (batch, output_len, dim) if provided"
        return self._forward(x, horizon, n_samples, y)
    
    @abstractmethod
    def _forward(self, x, horizon, n_samples, y=None):
        """
        Abstract method to be implemented by subclasses for the forward pass.

        Args:
            x (Tensor): Input sequence.
            horizon (int): Forecasting horizon.
            n_samples (int): Number of samples to generate.
            y (Tensor, optional): Teacher-forced target sequence.

        Returns:
            Tensor: Output of the model.
        """
        raise NotImplementedError("Subclasses must implement this method")

