"""
    TARGET:         Entire model
    PATCHES:        256 pixels 
    LOSS:           SSIM (non-negative) only
    INTENSITY:      0.2 
    EPOCHS:         250
    PREPROCESSING:  crop, flip and rotate

    Training 2023-09-08 17:17
"""

import script_config
import loss
from main import train
import preprocessing
import config as cfg

preprocessor = preprocessing.Unsharpen(patch_size=256,
                                       max_amount=0.2,
                                       rotate=True)

loss_fn = loss.SSIMLoss()

model_args = {
    "loss_fn": loss_fn,
    "learning_rate": 1e-3,
    "freeze_encoder": False,
    "use_espcn": True,
    "use_espcn_activations": True,
    "avoid_deconv": True,
    "use_alpha": True,
    "double_image_size": False
}

train(
    model_path=cfg.CKP_PATH / "alpha_kolnet_ssim" /
    "epoch=231-valid_loss=0.013.ckpt",
    model_args=model_args,
    preprocessor=preprocessor,
    n_epochs=250,
    run_name="alpha_kolnet_ssim_finetune",
    unfreeze_model=True,
)
