"""
    STEP 2:
    Train encoder-decoder on GoPro dataset on patches of 512 pixels.
    200 epochs, perceptual loss, no alpha channel, static learning rate 3e-5

    Training:  TBD
"""

from gc import unfreeze
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
    "freeze_encoder": False,
    "use_espcn": True,
    "use_espcn_activations": True,
    "avoid_deconv": True,
    "use_alpha": False,
    "double_image_size": False,
    "metrics": metrics,
}

batch_size_train, batch_size_test, n_epochs = 8, 8, 200
patch_size: int = 512

train(run_name="kolnet_v2_gopro",
      unfreeze_model=True,
      patch_size=patch_size,
      dataset=Dataset.GOPRO,
      model_args=model_args,
      n_epochs=n_epochs,
      batch_size_train=batch_size_train,
      batch_size_test=batch_size_test)
