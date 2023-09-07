"""
    Finetune the encoder (decoder already trained for 250 epochs) on patches of 
    128 pixels.

    Training 2023-08-23 17:12
"""
import config as cfg
from main import train
import preprocessing

preprocessor = preprocessing.Unsharpen(patch_size=128, max_amount=0.1)

train(model_path=(cfg.CKP_PATH / "alpha_perceptual_kolnet" /
                  "epoch=236-valid_loss=0.023.ckpt"),
      preprocessor=preprocessor,
      n_epochs=250,
      run_name="alpha_perceptual_kolnet_finetune")
