from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
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
    ):
        self.horizon = horizon
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logging = logging
        self.model = None

    @abstractmethod
    def _create_model(self):
        pass

    def set_horizon(self, horizon):
        self.horizon = horizon
        if self.model is not None:
            self.model.horizon = horizon

    def set_n_samples(self, n_samples):
        self.horn_samplesizon = n_samples
        if self.n_samples is not None:
            self.model.n_samples = n_samples

    def _create_trainer(self, len_dataloader):
        logger = False
        if self.logging:
            logger = CSVLogger(save_dir="logs")
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=int(len_dataloader * 0.1),
            logger=logger,
            accelerator=self.accelerator,
        )

    def fit(self, train_dataset, val_dataset=None):
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
        self._create_trainer(len(train_dataloader))
        self._create_model()
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate(self, test_dataset):
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self._create_trainer(len(test_dataloader))
        return self.trainer.test(self.model, test_dataloader)

    def predict(self, pred_dataset):
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        self._create_trainer(len(pred_dataloader))
        pred = self.trainer.predict(
            self.model,
            dataloaders=pred_dataloader,
        )
        return torch.cat(pred)


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
