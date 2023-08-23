"""
    Finetune the encoder (decoder already trained for 250 epochs) on patches of 
    128 pixels.
"""

from pathlib import Path
from main import train
import preprocessing

train(model_path=(Path("checkpoints") / "alpha_perceptual_kolnet" /
                  "epoch=236-valid_loss=0.023.ckpt"),
      preprocessor=preprocessing.unsharpen,
      n_epochs=250,
      run_name="alpha_perceptual_kolnet_finetune")
