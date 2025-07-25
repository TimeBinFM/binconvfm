from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from binconvfm.utils.metrics import mase, crps


class BaseForecaster:
    def __init__(
        self,
        input_len: int,
        output_len: int,
        quantiles: list[float] = [(i + 1) / 10 for i in range(9)],
        n_samples: int = 1000,
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        accelerator: str = "cpu",
        enable_progress_bar: bool = True,
        logging: bool = False,
    ):
        self.input_len = input_len
        self.output_len = output_len
        self.quantiles = quantiles
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logging = logging
        self.model = None

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
        logger = False
        if self.logging:
            logger = CSVLogger(save_dir="logs")
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=int(len(train_dataloader) * 0.1),
            logger=logger,
            accelerator=self.accelerator,
        )
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
        return self.trainer.test(self.model, test_dataloader)

    def predict(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
        )
        return self.trainer.predict(
            self.model,
            dataloaders=dataloader,
        )


class BaseForecasterModule(LightningModule):
    def __init__(self, input_len, output_len, quantiles, n_samples, lr):
        """
        PyTorch Lightning module for LSTM forecasting.

        Args:
            input_len (int): Length of input sequence.
            output_len (int): Length of output sequence.
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            n_samples (int): Number of samples for probabilistic output.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_len = input_len
        self.output_len = output_len
        self.quantiles = quantiles
        self.n_samples = n_samples
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_seq, horizon, target_seq = batch
        pred_seq = self.model(input_seq, horizon)  # (batch, n_samples, output_len)
        metrics = {
            "mase": mase(pred_seq, target_seq),
            "crps": crps(pred_seq, target_seq, self.quantiles),
        }
        self.log_dict(metrics, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        input_seq, horizon, _ = batch
        return self.model(input_seq, horizon)
