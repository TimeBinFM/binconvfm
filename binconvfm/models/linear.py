from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from binconvfm.models.base import BaseForecaster


class LinearForecaster(BaseForecaster):
    def __init__(self, input_len, output_len, batch_size = 32, num_epochs = 10, lr = 0.001, accelerator = "cpu", enable_progress_bar = True, logging = False):
        super().__init__(input_len, output_len, batch_size, num_epochs, lr, accelerator, enable_progress_bar, logging)
    
    def _create_model(self):
        self.model = LinearModule(self.input_len, self.output_len, self.lr)


class LinearModule(LightningModule):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Linear(input_dim, output_dim)
    
    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.loss(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_seq, target_seq = batch
        return self.model(input_seq)
    
    def loss(self, batch: Tensor) -> Tensor:
        input_seq, target_seq = batch
        pred_seq = self.model(input_seq)
        loss = F.mse_loss(pred_seq, target_seq)
        return loss


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        return x
