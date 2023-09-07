"""
    TARGET:         decoder only (encoder pretrained on imagenet)
    PATCHES:        128 pixels
    LOSS:           SSIM only 
    INTENSITY:      0.2 
    EPOCHS:         250
    PREPROCESSING:  only crop (rotations create black images that can be used to cheat)

    Training 2023-09-07 18:37
"""

import config
import loss
from main import train
import preprocessing

preprocessor = preprocessing.Unsharpen(patch_size=128,
                                       max_amount=0.2,
                                       rotate=False)

loss_fn = loss.SSIMLoss(weight=0.5)

model_args = {
    "loss_fn": loss_fn,
    "learning_rate": 1e-3,
    "freeze_encoder": True,
    "use_espcn": True,
    "use_espcn_activations": True,
    "avoid_deconv": True,
    "use_alpha": True,
    "double_image_size": False
}

train(model_args=model_args,
      preprocessor=preprocessor,
      n_epochs=250,
      run_name="alpha_perceptual_kolnet_ssim")
