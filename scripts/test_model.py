""" Validate a model """
from lightning.pytorch.trainer import Trainer

import script_config
import config as cfg
from data import TripodDataModule
import preprocessing
from kolnet import Kolnet

patch_size: int = 1024
model_name: str = "kolnet_perceptual"
run_name: str = f"{model_name}_{patch_size}"
ckpt_name: str = "last.ckpt"

model: Kolnet = Kolnet.load_from_checkpoint(cfg.CKP_PATH / run_name / ckpt_name)

preprocessor = preprocessing.Unsharpen(patch_size=patch_size,
                                       max_amount=0.15,
                                       add_alpha_channel=False)

datamodule: TripodDataModule = TripodDataModule(
    sample_target_generator=preprocessor, batch_size_train=4, batch_size_test=8)

trainer: Trainer = Trainer(precision="32", logger=False, detect_anomaly=True)

trainer.validate(model=model, datamodule=datamodule)
