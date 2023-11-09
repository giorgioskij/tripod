"""
    TARGET:         Entire model, starting from finetuned on 256 pixel images
    PATCHES:        1024 pixels 
    LOSS:           SSIM (non-negative) only
    INTENSITY:      0.15
    EPOCHS:         500. Already done 250 training + 500 finetune + 500 512pixels
    PREPROCESSING:  crop, flip and rotate
    LEARNING RATE:  4e-5

    Training 2023-09-09 19:53
"""

import script_config
import loss
from main import train
import preprocessing
import config as cfg
from kolnet import Kolnet

preprocessor = preprocessing.Unsharpen(patch_size=1024,
                                       max_amount=0.15,
                                       rotate=True)

loss_fn = loss.SSIMLoss()

model = Kolnet.load_from_checkpoint(cfg.CKP_PATH / "alpha_kolnet_ssim_512" /
                                    "epoch=323-valid_loss=0.013.ckpt",
                                    learning_rate=4e-5)

train(
    model=model,
    preprocessor=preprocessor,
    n_epochs=500,
    run_name="alpha_kolnet_ssim_1024",
    batch_size_train=4,
    batch_size_test=8,
)
