from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import LightningDataModule
from binconvfm.models.base import BaseLightningModule
from binconvfm.models.binconv import BinConvModule
from binconvfm.models.lstm import LSTMModule
from binconvfm.datamodules import DummyDataModule, SinDataModule, GiftEvalDataModule

if __name__ == "__main__":
    LightningCLI(
        model_class=BaseLightningModule,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
