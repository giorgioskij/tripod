# Project: tripod

import warnings
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch import loggers
from torch import nn

from data import Dataset, TripodDataModule, show
from edsr import EDSR
from loss import TripodLoss
from res_unet import UResNet
from unet import PoolingStrategy, UNet

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*full_state_update.*")
torch.set_float32_matmul_precision("medium")


class Architecture(Enum):
    UNET = auto()
    RES_UNET = auto()
    EDSR = auto()


class Loss(Enum):
    MSE = nn.MSELoss()
    TRIPOD = TripodLoss()
    L1 = nn.L1Loss()


def load_model(architecture: Architecture, loss: Loss) -> L.LightningModule:
    if architecture == Architecture.UNET:
        return UNet(loss_fn=loss.value,
                    pooling_strategy=PoolingStrategy.conv,
                    bilinear_upsampling=True)

    elif architecture == Architecture.RES_UNET:
        return UResNet(loss_fn=loss.value)

    elif architecture == Architecture.EDSR:
        return EDSR(
            n_features=64,
            residual_scaling=1,
            n_resblocks=16,
            loss_fn=loss.value,
            learning_rate=1e-3,
        )


def load_datamodule(dataset: Dataset, demo: bool = True) -> TripodDataModule:
    # load data and model
    d: TripodDataModule = TripodDataModule(batch_size_train=4,
                                           batch_size_test=16,
                                           dataset=dataset)
    d.setup()
    if demo:
        b = d.demo_batch()
        show(b)
    return d


def train(model: L.LightningModule,
          datamodule: TripodDataModule,
          epochs: int = 1,
          version: int = 0):
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        logger=loggers.CSVLogger(save_dir="./", version=version),
        precision="16-mixed",
        detect_anomaly=True,
    )
    trainer.fit(model=model, datamodule=datamodule)
    return trainer


def demo_model(model: L.LightningModule,
               d: TripodDataModule,
               train: bool = False,
               sigmoid: bool = True,
               save_path: Optional[Path] = None):
    test_batch = d.demo_batch(train=train)
    output = model(test_batch[0].to(model.device))
    if sigmoid:
        output = torch.sigmoid(output)
    if save_path is not None:
        show(test_batch, save_path)
        show(output, save_path)
    else:
        show(test_batch)
        show(output)


if __name__ == "__main__":
    d = load_datamodule(Dataset.DIV2K)
    d.load_from_checkpoint(
        "lightning_logs/version_4/checkpoints/epoch=9-step=2000.ckpt")
    m = load_model(Architecture.EDSR, Loss.L1)
    trainer = train(m, d, epochs=10, version=4)
    demo_model(m, d, train=True)
