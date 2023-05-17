# Project: tripod

import os
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Type

import lightning as L
import torch
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import (Callback, ModelCheckpoint,
                                         OnExceptionCheckpoint)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.utilities.types import STEP_OUTPUT
from matplotlib import pyplot as plt
from torch import nn, Tensor, tensor

import edsr_paper
from data import Dataset, TripodDataModule, show, tensor_to_image
from edsr import EDSR
from loss import TripodLoss
from res_unet import UResNet
from unet import PoolingStrategy, UNet

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*full_state_update.*")
warnings.filterwarnings("ignore", ".*exists and is non empty.*")
warnings.filterwarnings("ignore", ".*attribute 'loss_fn' is an instance.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")
warnings.filterwarnings("ignore", ".*Clipping input data.*")

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
               save_path: Optional[Path] = None):
    test_batch = d.demo_batch(train=train)
    output = model(test_batch[0].to(model.device))
    if model.uses_sigmoid:
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


# pytorch lightning callback that saves a batch of images every 5 epochs
class SaveImages(Callback):

    def __init__(self, save_path: Path, every_n_epochs: int = 1):
        super().__init__()
        self.save_path = save_path
        self.every_n_epochs: int = every_n_epochs

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: Tensor,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:

        if batch_idx != 0:
            return

        if trainer.current_epoch % self.every_n_epochs == 0:
            os.makedirs(self.save_path, exist_ok=True)
            save_path = (
                self.save_path /
                f"epoch={trainer.current_epoch}-step={trainer.global_step}")
            predictions = pl_module(batch[0][:4].to(pl_module.device))
            f, ax = plt.subplots(3, 4, figsize=(20, 20))
            for i, (output, sample,
                    target) in enumerate(zip(predictions, batch[0], batch[1])):
                if i > 3:
                    break
                img1, img2, img3 = (tensor_to_image(sample),
                                    tensor_to_image(output),
                                    tensor_to_image(target))
                ax[0, i].imshow(img1)
                ax[1, i].imshow(img2)
                ax[2, i].imshow(img3)
            plt.savefig(save_path)


def test_model(model_class: Type, ckp_path: Path | str):
    m = model_class.load_from_checkpoint(str(ckp_path), loss_fn=nn.L1Loss)
    d = TripodDataModule(Dataset.DIV2K, batch_size_train=32, batch_size_test=64)
    d.setup()
    demo_model(m, d, train=False)
    return m, d


def train_model():
    logger = loggers.WandbLogger(project="tripod")
    ckp_path = "checkpoints/unet_residual_blocks/"
    ckp = ModelCheckpoint(
        dirpath=ckp_path,
        save_top_k=2,
        monitor="valid_loss",
        filename="{epoch}-{valid_loss:.3f}",
    )
    int_ckp = OnExceptionCheckpoint(dirpath=ckp_path, filename="interrupted")

    save_images_callback = SaveImages(save_path=Path(ckp_path) / "images",
                                      every_n_epochs=5)

    d = TripodDataModule(Dataset.DIV2K, batch_size_train=16, batch_size_test=64)

    # m = EDSR(n_features=64,
    #          residual_scaling=0.5,
    #          n_resblocks=16,
    #          loss_fn=nn.L1Loss(),
    #          learning_rate=4e-5)
    m = UNet(
        loss_fn=nn.L1Loss(),
        pooling_strategy=PoolingStrategy.conv,
        residual=True,
        bilinear_upsampling=False,
        learning_rate=1e-4,
    )
    trainer = L.Trainer(
        # default_root_dir=ckp_path,
        callbacks=[ckp, int_ckp, save_images_callback],
        accelerator="auto",
        max_epochs=1000,
        logger=logger,
        precision="16-mixed",
        # gradient_clip_val=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
    )
    trainer.fit(m, d)
    # trainer.fit(
    #     m,
    #     datamodule=d,
    #     # ckpt_path="last",
    #     # ckpt_path=str(Path(ckp_path) / "epoch=25-valid_loss=0.026.ckpt"),
    #     # ckpt_path="checkpoints/edsr/interrupted.ckpt",
    # )

    # d.setup()
    # demo_model(m, d, train=False)
    # tune_params(m, d)


def train_paper_edsr():
    logger = loggers.WandbLogger(project="tripod")
    ckp_path = "checkpoints/edsr-paper"
    ckp = ModelCheckpoint(
        dirpath=ckp_path,
        save_top_k=2,
        monitor="valid_loss",
        filename="{epoch}-{valid_loss:.3f}",
    )
    int_ckp = OnExceptionCheckpoint(dirpath=ckp_path, filename="interrupted")

    d = TripodDataModule(Dataset.DIV2K,
                         batch_size_train=32,
                         batch_size_test=64,
                         sample_patch_size=48,
                         target_patch_size=96)
    m = edsr_paper.EDSRLightning()

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
    trainer.fit(model=m,
                datamodule=d,
                ckpt_path=str(Path(ckp_path) / "epoch=9-valid_loss=1.546.ckpt"))

    # d.setup()
    # demo_model(m, d, train=False)
    # tune_params(m, d)


if __name__ == "__main__":

    # test_model(model_class=UNet,
    #            ckp_path="checkpoints/unet/epoch=605-valid_loss=0.016.ckpt")

    # m, d = test_model(
    #     model_class=EDSR,
    #     ckp_path="checkpoints/edsr/epoch=891-valid_loss=0.082.ckpt")

    # train_paper_edsr()

    train_model()
