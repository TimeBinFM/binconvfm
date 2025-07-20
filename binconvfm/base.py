from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from abc import abstractmethod


class BaseForecaster():
    def __init__(
            self,
            input_len: int,
            output_len: int,
            batch_size: int = 32,
            num_epochs: int = 10,
            lr: float = 0.001,
            accelerator: str = "cpu",
            enable_progress_bar: bool = True,
            logging: bool = False
        ):
        self.input_len = input_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logging = logging
        self.model = None
        self._create_model()

    @abstractmethod
    def _create_model(self):
        pass

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
            logger = CSVLogger(save_dir='logs')
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=int(len(train_dataloader)*0.1),
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
        return self.trainer.test(
            self.model, 
            test_dataloader
        )
        
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
