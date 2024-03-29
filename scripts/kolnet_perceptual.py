""" First training phase on encoder only, patches of 256 pixels
    Then finetune full model with patches of 512 pixels
    Finally 1024 pixels

    No alpha, and ssim only as a metric, use perceptual loss + mse to train

    Training 2023-11-09 12:44
"""

import script_config

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
model = None
for i, patch_size in enumerate((256, 512, 1024)):
    preprocessor = preprocessing.Unsharpen(patch_size=patch_size,
                                           max_amount=0.15,
                                           flip=True,
                                           rotate=True,
                                           add_alpha_channel=False)
    unfreeze_model: bool = (patch_size != 256)

    if patch_size == 256:
        batch_size_train, batch_size_test, n_epochs = 32, 64, 400
    elif patch_size == 512:
        batch_size_train, batch_size_test, n_epochs = 8, 8, 200
    else:
        batch_size_train, batch_size_test, n_epochs = 4, 4, 100

    model = train(
        preprocessor,
        model=model,
        precision="32",
        model_args=model_args,
        n_epochs=n_epochs,
        run_name=f"kolnet_perceptual_{patch_size}",
        unfreeze_model=unfreeze_model,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
    )

    torch.cuda.empty_cache()
