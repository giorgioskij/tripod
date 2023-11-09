import script_config
import config as cfg
from PIL import Image
import loss
import preprocessing
from kolnet import Kolnet
from data import show
import torch

preprocessor = preprocessing.Unsharpen(patch_size=1024,
                                       max_amount=0.15,
                                       rotate=True)

loss_fn = loss.SSIMLoss()

model = Kolnet.load_from_checkpoint(cfg.CKP_PATH / "alpha_kolnet_ssim_1024" /
                                    "epoch=208-valid_loss=0.067.ckpt",
                                    learning_rate=4e-5)

TEST_IMAGE_PATH = "./datasets/DIV2K/DIV2K_valid_HR/0803.png"
real_amount = 0.5
im = Image.open(TEST_IMAGE_PATH)
im_sample, im_target = preprocessor(im, amount=real_amount)
im_sample = im_sample[:3, ...]
print(f"Sample image was blurred by a factor of {real_amount}")
show(im_sample)

outputs = []
titles = []
for amount in range(0, 11):
    amount /= 10
    outputs.append(
        model.sharpen(im_sample, amount=amount).detach().cpu().squeeze())
    titles.append(f"Amount: {amount}")

outputs = torch.stack(outputs, dim=0)
show(outputs, title=titles)
