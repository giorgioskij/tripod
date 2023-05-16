# Project: tripod

import warnings
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from lightning.pytorch import loggers
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from torch import nn

from data import Dataset, TripodDataModule, show
from edsr import EDSR
from loss import TripodLoss
from res_unet import UResNet
from unet import PoolingStrategy, UNet

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*full_state_update.*")
warnings.filterwarnings("ignore", ".*exists and is non empty.*")
torch.set_float32_matmul_precision("medium")


class Architecture(Enum):
    UNET = auto()
    RES_UNET = auto()
    EDSR = auto()


class Loss(Enum):
    MSE = nn.MSELoss()
    TRIPOD = TripodLoss()
    L1 = nn.L1Loss()


def load_model(architecture: Architecture,
               loss: Loss,
               lr: float = 1e-3) -> L.LightningModule:
    if architecture == Architecture.UNET:
        return UNet(loss_fn=loss.value,
                    pooling_strategy=PoolingStrategy.conv,
                    bilinear_upsampling=True)

    elif architecture == Architecture.RES_UNET:
        return UResNet(loss_fn=loss.value)

    elif architecture == Architecture.EDSR:
        return EDSR(
            n_features=64,
            # residual_scaling=0.1,
            n_resblocks=16,
            loss_fn=loss.value,
            learning_rate=lr,
        )


def load_datamodule(dataset: Dataset,
                    demo: bool = True,
                    batch_size_train: int = 32) -> TripodDataModule:
    # load data and model
    d: TripodDataModule = TripodDataModule(batch_size_train=batch_size_train,
                                           batch_size_test=64,
                                           dataset=dataset)
    d.setup()
    if demo:
        print("Preview of dataset. Row 1: samples, Row 2: targets")
        b = d.demo_batch()
        show(b)
    return d


def train(
    model: L.LightningModule,
    datamodule: TripodDataModule,
    root_dir: str | Path,
    epochs: int = 1,
    version: int = 0,
    restart_from: Optional[str | Path] = None,
):
    trainer = L.Trainer(
        default_root_dir=root_dir,
        accelerator="auto",
        max_epochs=epochs,
        logger=loggers.CSVLogger(save_dir="./", version=version),
        precision="16-mixed",
        detect_anomaly=True,
        gradient_clip_val=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=0,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(restart_from))
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


def tune_params(m: L.LightningModule, d: TripodDataModule):

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=1,
        logger=None,
        precision="16-mixed",
        gradient_clip_val=1,
    )
    tuner = Tuner(trainer)

    # find optimal learning rate
    # tuner.lr_find(model=m, datamodule=d)

    # find maximum batch size for training
    tuner.scale_batch_size(model=m,
                           datamodule=d,
                           method="fit",
                           mode="binsearch")
    # find maximum batch size for validation
    tuner.scale_batch_size(model=m,
                           datamodule=d,
                           method="validate",
                           mode="binsearch")


if __name__ == "__main__":

    logger = loggers.WandbLogger(project="tripod")
    ckp = ModelCheckpoint(
        dirpath="checkpoints/unet/",
        save_top_k=2,
        monitor="valid_loss",
        filename="{epoch}-{valid_loss:.3f}",
    )
    int_ckp = OnExceptionCheckpoint(dirpath="checkpoints/unet/",
                                    filename="interrupted")

    d = TripodDataModule(Dataset.DIV2K, batch_size_train=32, batch_size_test=64)

    # m = EDSR(n_features=64,
    #          residual_scaling=0.5,
    #          n_resblocks=16,
    #          loss_fn=nn.L1Loss(),
    #          learning_rate=4e-5)
    m = UNet(loss_fn=nn.L1Loss(),
             pooling_strategy=PoolingStrategy.conv,
             bilinear_upsampling=True)

    trainer = L.Trainer(
        callbacks=[ckp, int_ckp],
        accelerator="auto",
        max_epochs=1000,
        logger=logger,
        precision="16-mixed",
        detect_anomaly=True,
        gradient_clip_val=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
    )
    trainer.fit(
        model=m,
        datamodule=d,
        # ckpt_path="checkpoints/edsr/interrupted.ckpt",
    )

    # d.setup()
    # demo_model(m, d, train=True)
    # tune_params(m, d)
