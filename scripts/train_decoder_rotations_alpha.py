"""
    Train the decoder only on patches of 128 pixels with higher unsharpen 
    intensity (0.2) and random rotations.

    Also, the perceptual loss is given a lower weigtht (0.2 instead of 0.5)

    Training 2023-09-07 16:48
"""

import config as cfg
from loss import TripodLoss
from main import train
import preprocessing

preprocessor = preprocessing.Unsharpen(patch_size=128, max_amount=0.2)

perceptual_loss = TripodLoss(weight=0.2)
model_args = {
    "loss_fn": perceptual_loss,
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
      run_name="alpha_perceptual_kolnet_harder")
