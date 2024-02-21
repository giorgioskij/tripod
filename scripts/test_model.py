""" Validate a model """
# from lightning.pytorch.trainer import Trainer
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

import script_config
import config as cfg
from data import TripodDataModule
import preprocessing
from kolnet import Kolnet

patch_size: int = 1024
model_name: str = "kolnet_perceptual"
run_name: str = f"{model_name}_{patch_size}"
ckpt_name: str = "epoch=386-valid/global_loss=0.042.ckpt"

model: Kolnet = Kolnet.load_from_checkpoint(cfg.CKP_PATH / run_name /
                                            ckpt_name).eval().cuda()

preprocessor = preprocessing.Unsharpen(patch_size=1024,
                                       flip=False,
                                       rotate=False,
                                       pad=False,
                                       max_amount=0.15,
                                       add_alpha_channel=False)
# image = Image.open("/home/tkol/Desktop/boots1024.jpg").convert("RGB")
# blurred, _ = preprocessor(image, amount=0.2)
# save_image(blurred, "/home/tkol/Desktop/blurred_boots.jpg")
# output = model.sharpen(blurred)
# save_image(output, "/home/tkol/Desktop/sharpened_boots.jpg")

datamodule: TripodDataModule = TripodDataModule(
    sample_target_generator=preprocessor, batch_size_train=4, batch_size_test=8)
datamodule.setup()

# trainer: Trainer = Trainer(precision="32", logger=False, detect_anomaly=True)

# trainer.validate(model=model, datamodule=datamodule)

valid_dataloader = datamodule.val_dataloader()
bs = valid_dataloader.batch_size
for i, (inputs, targets) in tqdm(enumerate(valid_dataloader),
                                 total=len(valid_dataloader)):

    with torch.no_grad():
        outputs = model(inputs.cuda())

    for j in range(inputs.size(0)):
        idx = i * bs + j  # type: ignore
        save_image(inputs[j].cpu(), f"/home/tkol/Desktop/inputs/{idx}.jpg")
        save_image(targets[j], f"/home/tkol/Desktop/targets/{idx}.jpg")
        save_image(outputs[j].detach().cpu(),
                   f"/home/tkol/Desktop/outputs/{idx}.jpg")
