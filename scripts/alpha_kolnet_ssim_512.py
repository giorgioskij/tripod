"""
    TARGET:         Entire model, starting from finetuned on 256 pixel images
    PATCHES:        512 pixels 
    LOSS:           SSIM (non-negative) only
    INTENSITY:      0.15
    EPOCHS:         500. Already done 250 training + 500 finetune
    PREPROCESSING:  crop, flip and rotate
    LEARNING RATE:  1e-4

    Training 2023-09-08 20:21
"""

import script_config
import loss
from main import train
import preprocessing
import config as cfg
from uresnet import UResNet

preprocessor = preprocessing.Unsharpen(patch_size=512,
                                       max_amount=0.15,
                                       rotate=True)

loss_fn = loss.SSIMLoss()

model = UResNet.load_from_checkpoint(cfg.CKP_PATH /
                                     "alpha_kolnet_ssim_finetune" /
                                     "epoch=419-valid_loss=0.000.ckpt",
                                     learning_rate=1e-4)

train(
    model=model,
    preprocessor=preprocessor,
    n_epochs=500,
    run_name="alpha_kolnet_ssim_512",
    batch_size_train=16,
    batch_size_test=16,
)
