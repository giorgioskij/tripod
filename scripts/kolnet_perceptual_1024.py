""" 
    Train full model on patches of 1024 pixels 

    No alpha, and ssim only as a metric, use perceptual loss + mse to train

    Training 2023-11-10 16:53
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import script_config
import config as cfg

from torch import nn
import torch

import preprocessing
import loss
from main import train

perceptual = loss.PerceptualLoss(weight=0.5)
ssim = loss.SSIMLoss()
metrics = nn.ModuleDict({"ssim": ssim})
# metrics = nn.ModuleDict({"perceptual": perceptual})

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
patch_size: int = 1024
preprocessor = preprocessing.Unsharpen(patch_size=patch_size,
                                       max_amount=0.15,
                                       flip=True,
                                       rotate=True,
                                       add_alpha_channel=False)

batch_size_train, batch_size_test, n_epochs = 4, 4, 400

train(
    preprocessor,
    model_path=cfg.CKP_PATH / "kolnet_perceptual_512" / "last.ckpt",
    precision="32",
    model_args=model_args,
    n_epochs=n_epochs,
    run_name=f"kolnet_perceptual_{patch_size}",
    unfreeze_model=True,
    batch_size_train=batch_size_train,
    batch_size_test=batch_size_test,
)
