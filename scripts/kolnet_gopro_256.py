"""
    Train decoder only on GoPro dataset on patches of 256 pixels.
    400 epochs, perceptual loss, no alpha channel, static learning rate 3e-5

    Training 2024-02-23 18:17
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import script_config
import config as cfg

from torch import nn

import loss
from main import train
from data import Dataset

perceptual = loss.PerceptualLoss(weight=0.5)
ssim = loss.SSIMLoss()
metrics = nn.ModuleDict({"ssim": ssim})

loss_fn = perceptual

model_args = {
    "loss_fn": loss_fn,
    "learning_rate": 3e-5,
    "freeze_encoder": True,
    "use_espcn": True,
    "use_espcn_activations": True,
    "avoid_deconv": True,
    "use_alpha": False,
    "double_image_size": False,
    "metrics": metrics,
}
patch_size: int = 256

batch_size_train, batch_size_test, n_epochs = 16, 16, 400

train(run_name="kolnet_v2_gopro",
      dataset=Dataset.GOPRO,
      model_args=model_args,
      n_epochs=n_epochs,
      batch_size_train=batch_size_train,
      batch_size_test=batch_size_test)
