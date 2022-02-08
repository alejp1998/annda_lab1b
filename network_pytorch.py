# KTH DD2437 Lab 1B Filip Geib 2022
import numpy as np
import torch
from torch import nn
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

import matplotlib.pyplot as plt
from data_generator import Experiment


class DataModule(pl.LightningDataModule):
    def __init__(self, exp_name, batch_size):
        super().__init__()
        # load experiment data
        self.exp = Experiment('')
        self.exp.load('data', exp_name)

        self.batch_size = batch_size
        self.series = self.exp.series

    def setup(self, stage: Optional[str] = None):
        self.data_tra = self.create_tensor_dataset(self.exp.tra_pts, self.exp.tra_lab)
        self.data_val = self.create_tensor_dataset(self.exp.val_pts, self.exp.val_lab)
        self.data_tes = self.create_tensor_dataset(self.exp.tes_pts, self.exp.tes_lab)

    @staticmethod
    def create_tensor_dataset(data_x, data_y):
        tensor_x = torch.from_numpy(data_x).float()
        tensor_y = torch.from_numpy(data_y).float()
        return TensorDataset(tensor_x, tensor_y)

    def train_dataloader(self):
        return DataLoader(self.data_tra, batch_size=self.batch_size, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.data_tes, batch_size=self.batch_size, num_workers=16)
# end class DataModule


class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
        self.ce = nn.MSELoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.ce(y_hat, y)
        self.log('tra_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.ce(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.ce(y_hat, y)
        self.log('tes_loss', loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=1000,
        # cycle_momentum=False)
        return optimizer    #, [scheduler]
# end class MLP


if __name__ == "__main__":
    # setup parameters
    TRAIN = False
    EPOCHS = 1000
    DEVICE = 'gpu'
    GPU = 1 if DEVICE == 'gpu' else 0
    NAME = 'milf_task2_5to6_015noise'
    PLTSTR = 'noise sigma: 0.15, layers: 5x6'

    # fix RNG seed
    pl.seed_everything(42)

    # load timeseries dataset and construct datasets
    data = DataModule('exp_pytorch_015noise', batch_size=16)

    # define logger
    logger = pl.loggers.TensorBoardLogger("logs/", version=NAME)

    # train network
    if TRAIN:
        model = MLP()

        # define callbacks
        es_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=32, mode='min')
        lr_callback = LearningRateMonitor(logging_interval='step')

        # define trainer
        trainer = pl.Trainer(callbacks=[es_callback, lr_callback], max_epochs=EPOCHS, logger=logger, gpus=GPU)
        trainer.fit(model, data)

        torch.save(model.state_dict(), 'models/{}'.format(NAME))

    # test network
    else:
        model = MLP()
        model.load_state_dict(torch.load('models/{}'.format(NAME)))

        # define trainer
        trainer = pl.Trainer(logger=logger, gpus=GPU)
        trainer.test(model, data)

        # compare with clean data
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))

        series = np.zeros(1200)
        for i in range(300, 1500 - 10):
            tmp_sample = data.series[[i - 20, i - 15, i - 10, i - 5, i]]
            tmp_sample = torch.from_numpy(tmp_sample).float()
            series[i - 300] = model.forward(tmp_sample)

        # plot overlap with series
        ax.plot(range(1200), data.series[300:], label='Original')
        ax.plot(range(1200), series, '-.', label='Predicted')
        ax.legend(ncol=2)
        ax.grid()
        #ax.set_xlim([1000, 1200])
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(PLTSTR)

        plt.tight_layout()

        # reconstruct time series
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))

        series = np.zeros(1500)
        series[300] = 1.5

        for i in range(300, 1500 - 10):
            sample = series[[i - 20, i - 15, i - 10, i - 5, i]]
            sample = torch.from_numpy(sample).float()
            series[i + 5] = model.forward(sample)

        # plot overlap with series
        ax.plot(range(1200), data.series[300:], label='Original')
        ax.plot(range(5, 1205), series[300:], '-.', label='Predicted')
        ax.legend(ncol=2)
        ax.grid()
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.set_title(PLTSTR)

        plt.tight_layout()
        plt.show()
