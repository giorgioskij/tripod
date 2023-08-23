"""
    Finetune the encoder (decoder already trained for 250 epochs) on patches of 
    128 pixels.
"""
import sys

sys.path.append("../")

import config as cfg
from main import train
import preprocessing

train(model_path=(cfg.CKP_PATH / "alpha_perceptual_kolnet" /
                  "epoch=236-valid_loss=0.023.ckpt"),
      preprocessor=preprocessing.unsharpen,
      n_epochs=1,
      run_name="alpha_perceptual_kolnet_finetune")
