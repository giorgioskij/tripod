# Project: tripod
import config as cfg

import os
import warnings
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Dict, Callable

import lightning as L
import torch
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import (Callback, ModelCheckpoint,
                                         OnExceptionCheckpoint)
from lightning.pytorch.tuner import Tuner
from matplotlib import pyplot as plt
from torch import nn, Tensor
import wandb
from PIL import Image

from data import Dataset, TripodDataModule, show, tensor_to_image
from preprocessing import tripod_transforms
from edsr import EDSR
from kolnet import Kolnet
from loss import PerceptualLoss
from kolnet import Kolnet
from unet import PoolingStrategy, UNet
import preprocessing

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*full_state_update.*")
warnings.filterwarnings("ignore", ".*exists and is non empty.*")
warnings.filterwarnings("ignore", ".*attribute 'loss_fn' is an instance.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")

torch.set_float32_matmul_precision("medium")

plt.switch_backend('agg')


class Architecture(Enum):
    UNET = auto()
    RES_UNET = auto()
    EDSR = auto()


class Loss(Enum):
    MSE = nn.MSELoss()
    TRIPOD = PerceptualLoss()
    L1 = nn.L1Loss()


def load_model(architecture: Architecture,
               loss: Loss,
               lr: float = 1e-3) -> L.LightningModule:
    if architecture == Architecture.UNET:
        return UNet(loss_fn=loss.value,
                    pooling_strategy=PoolingStrategy.conv,
                    bilinear_upsampling=True)

    elif architecture == Architecture.RES_UNET:
        return Kolnet(loss_fn=loss.value)

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


# def train(
#     model: L.LightningModule,
#     datamodule: TripodDataModule,
#     root_dir: str | Path,
#     epochs: int = 1,
#     version: int = 0,
#     restart_from: Optional[str | Path] = None,
# ):
#     trainer = L.Trainer(
#         default_root_dir=root_dir,
#         accelerator="auto",
#         max_epochs=epochs,
#         logger=loggers.CSVLogger(save_dir="./", version=version),
#         precision="16-mixed",
#         detect_anomaly=True,
#         gradient_clip_val=1,
#         check_val_every_n_epoch=1,
#         log_every_n_steps=0,
#     )
#     trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(restart_from))
#     return trainer


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
        # gradient_clip_val=1,
    )
    tuner = Tuner(trainer)

    # find optimal learning rate
    tuner.lr_find(model=m, datamodule=d)

    # find maximum batch size for training
    # tuner.scale_batch_size(model=m,
    #                        datamodule=d,
    #                        method="fit",
    #                        mode="binsearch")
    # find maximum batch size for validation
    # tuner.scale_batch_size(model=m,
    #                        datamodule=d,
    #                        method="validate",
    #                        mode="binsearch")


# pytorch lightning callback that saves a batch of images every 5 epochs
class SaveImages(Callback):

    def __init__(self,
                 save_path: Path,
                 every_n_epochs: int = 1,
                 log_every: Optional[int] = None):
        super().__init__()
        self.save_path = save_path
        self.every_n_epochs: int = every_n_epochs
        self.log_every: Optional[int] = log_every

    def on_validation_batch_end(self,
                                trainer: L.Trainer,
                                pl_module: L.LightningModule,
                                outputs: Tensor,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:

        if batch_idx != 0:
            return

        if trainer.global_step == 0 or (trainer.current_epoch +
                                        1) % self.every_n_epochs == 0:
            os.makedirs(self.save_path, exist_ok=True)
            save_path = (
                self.save_path /
                f"epoch={trainer.current_epoch}-step={trainer.global_step}")
            predictions = pl_module(batch[0][:4].to(pl_module.device))
            f, ax = plt.subplots(3, 4, figsize=(20, 20))

            columns = ["input", "output", "ground truth"]
            data = []
            for i, (output, sample,
                    target) in enumerate(zip(predictions, batch[0], batch[1])):
                if i > 3:
                    break
                row = []
                for x, image in enumerate([sample, output, target]):
                    if image.shape[-3] == 4:
                        image = image[..., :3, :, :]
                    image = tensor_to_image(image)
                    ax[x, i].imshow(image)
                    row.append(wandb.Image(image))
                data.append(row)

                # img1, img2, img3 = (tensor_to_image(sample),
                #                     tensor_to_image(output),
                #                     tensor_to_image(target))
                # ax[0, i].imshow(img1)
                # ax[1, i].imshow(img2)
                # ax[2, i].imshow(img3)

            plt.savefig(save_path)
            plt.close()

            if (self.log_every and
                    trainer.current_epoch % self.log_every == 0 and
                    pl_module.logger is not None):
                pl_module.logger.log_table(  #type:ignore
                    key=f"demos_step{trainer.global_step}",
                    columns=columns,
                    data=data)


# def test_model(model_class: Type, ckp_path: Path | str):
#     m = model_class.load_from_checkpoint(str(ckp_path), loss_fn=nn.L1Loss)
#     d = TripodDataModule(Dataset.DIV2K, batch_size_train=32, batch_size_test=64)
#     d.setup()
#     demo_model(m, d, train=False)
#     return m, d


def setup_trainer(
        n_epochs: int = 1,
        save_images_every: int = 10,
        log_images_every: int = 50,
        precision: str = "32",
        #   log: bool = True,
        run_name: Optional[str] = None) -> L.Trainer:

    output_dir: Path = Path("checkpoints") / (run_name or "unnamed_run")

    logger = (loggers.WandbLogger(project="tripod", name=run_name)
              if run_name else False)

    # if ckp_path is None and run_name is None:
    #     raise ValueError("Either ckp_path or run_name must be specified")
    # if ckp_path is None and run_name is not None:
    #     ckp_path = Path("checkpoints") / run_name
    # if run_name is None and ckp_path is not None:
    #     run_name = ckp_path.name

    ckp = ModelCheckpoint(
        # dirpath=ckp_path,
        dirpath=output_dir,
        save_top_k=2,
        monitor="valid_loss",
        filename="{epoch}-{valid_loss:.3f}",
        save_last=True,
    )
    int_ckp = OnExceptionCheckpoint(
        dirpath=output_dir,  #type:ignore
        filename="interrupted")
    callbacks = [ckp, int_ckp]

    if save_images_every > 0:
        save_images_callback = SaveImages(
            save_path=output_dir / "images",  #  type:ignore
            every_n_epochs=save_images_every,
            log_every=log_images_every)
        callbacks.append(save_images_callback)

    trainer = L.Trainer(
        callbacks=callbacks,
        accelerator="auto",
        max_epochs=n_epochs,
        logger=logger,
        precision=precision,  #type: ignore
        gradient_clip_val=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        # num_sanity_val_steps=0,
    )
    return trainer


def hyperparam_sweep():
    epochs = 200

    # freeze encoder, train decoder: L1 loss, no espcn
    # try:
    #     m = UResNet(loss_fn=nn.L1Loss(), learning_rate=1e-4, use_espcn=False)
    #     trainer = setup_trainer(n_epochs=epochs, run_name="k_l1_noespcn")
    #     d = TripodDataModule()
    #     trainer.fit(model=m, datamodule=d)
    #     wandb.finish()
    # except:
    #     print("---------- Run failed ----------")

    # same, but with l2 loss
    # try:
    #     m = UResNet(loss_fn=nn.MSELoss(), learning_rate=1e-4, use_espcn=False)
    #     trainer = setup_trainer(n_epochs=epochs, run_name="k_l2_noespcn")
    #     d = TripodDataModule()
    #     trainer.fit(model=m, datamodule=d)
    #     wandb.finish()
    # except:
    #     print("---------- Run failed ----------")

    # same, but with espcn and l2 loss
    try:
        m = Kolnet(loss_fn=nn.MSELoss(), learning_rate=1e-4, use_espcn=True)
        trainer = setup_trainer(n_epochs=epochs, run_name="k_l2_espcn")
        d = TripodDataModule()
        trainer.fit(model=m, datamodule=d)
        wandb.finish()
    except:
        print("---------- Run failed ----------")

    # same, but with espcn and l2 loss but no activations
    # try:
    #     m = UResNet(loss_fn=nn.L1Loss(), learning_rate=1e-4, use_espcn=True)
    #     trainer = setup_trainer(n_epochs=epochs, run_name="k_l2_espcn_noact")
    #     d = TripodDataModule()
    #     trainer.fit(model=m, datamodule=d)
    #     wandb.finish()
    # except:
    #     print("---------- Run failed ----------")

    # same, but with larger learning rate
    # try:
    #     trainer = setup_trainer(n_epochs=epochs, run_name="k_l1_noespcn_lr1e-3")
    #     m = UResNet(loss_fn=nn.L1Loss(), learning_rate=1e-3, use_espcn=False)
    #     d = TripodDataModule()
    #     trainer.fit(model=m, datamodule=d)
    #     wandb.finish()
    # except:
    #     print("---------- Run failed ----------")


def test_model():
    checkpoint: Path = cfg.CKP_PATH / "alpha_perceptual_kolnet_finetune" / "last.ckpt"
    output_path = None

    preprocessor = preprocessing.Unsharpen()
    d = TripodDataModule(sample_target_generator=preprocessor)
    d.setup()
    m = Kolnet.load_from_checkpoint(checkpoint,
                                    map_location=torch.device("cpu"))
    # trainer = L.Trainer(precision="16-mixed", logger=False)
    # trainer.validate(model=m, datamodule=d)

    b = d.demo_batch(train=False)
    with torch.no_grad():
        pred = m(b[0].to(m.device))
    show(b)
    show(pred)
    return m

    # images = [
    #     Image.open(output_path / f"transformed/img{i}.png") for i in range(4)
    # ]
    # resampled = [
    #     i.resize((128, 128), resample=Image.Resampling.BILINEAR) for i in images
    # ]
    # for i, image in enumerate(resampled):
    #     image.save(output_path / f"transformed/resampled{i}.png")


if __name__ == "__main__":
    # m = test_model()
    # epochs = 250
    pass

    # m = UResNet(loss_fn=TripodLoss(),
    #             learning_rate=1e-4,
    #             use_espcn=True,
    #             avoid_deconv=True,
    #             use_alpha=True,
    #             double_image_size=False,
    #             freeze_encoder=False)

    # trainer = setup_trainer(n_epochs=epochs, run_name="alpha_perceptual_kolnet")
    # d = TripodDataModule(sample_target_generator=preprocessing.unsharpen)
    # trainer.fit(model=m, datamodule=d)

    # train encoder and decoder
    # trainer = setup_trainer(
    #     Path("checkpoints/kolnet-finetune"),
    #     save_images_every=10,
    #     log=True,
    #     n_epochs=500,
    # )
    # m = UResNet.load_from_checkpoint("ckeckpoints/kolnet/last.ckpt",
    #                                  freeze_encoder=False)
    # d = TripodDataModule(Dataset.DIV2K,
    #                      batch_size_train=16,
    #                      batch_size_test=64,
    #                      sample_patch_size=64,
    #                      target_patch_size=128)
    # trainer.fit(model=m, datamodule=d)


def train(
    preprocessor: Callable,
    model_path: Optional[Path] = None,
    model_args: Optional[Dict] = None,
    model: Optional[Kolnet] = None,
    precision: str = "32",
    n_epochs: int = 1,
    run_name: Optional[str] = None,
    unfreeze_model: bool = False,
    batch_size_train: int = 16,
    batch_size_test: int = 64,
):

    if model is not None:
        m = model
    elif model_path is not None:
        m = Kolnet.load_from_checkpoint(model_path)
    elif model_args is not None:
        m = Kolnet(**model_args)
    else:
        raise ValueError("Either model args or model path must be specified")

    if unfreeze_model:
        for p in m.parameters():
            p.requires_grad = True

    trainer = setup_trainer(n_epochs=n_epochs,
                            run_name=run_name,
                            precision=precision)
    d = TripodDataModule(sample_target_generator=preprocessor,
                         batch_size_train=batch_size_train,
                         batch_size_test=batch_size_test)
    trainer.fit(model=m, datamodule=d)

    wandb.finish()

    return m
