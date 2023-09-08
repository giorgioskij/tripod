import numpy as np
import torch
from torch import nn
from pytorch_msssim import SSIM, MS_SSIM
from torchvision import models
from torch import Tensor
from torchinfo import summary

from data import TripodDataModule
from PIL import Image
from matplotlib import pyplot as plt
import preprocessing


class PerceptualLoss(nn.Module):

    def __init__(self, weight: float = 0.5):
        super().__init__()
        # self.ssim = SSIM(data_range=1, nonnegative_ssim=True)
        self.weight: float = weight
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.vgg.eval()

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:

        # VGG perceptual loss
        self.vgg.to(X.device)
        with torch.no_grad():
            target_vgg = self.vgg(Y)
            x_vgg = self.vgg(X)
        feature_loss = self.mae(x_vgg, target_vgg)

        pixel_loss = self.mse(X, Y)  # pixel loss

        loss = feature_loss * self.weight + pixel_loss * (1 - self.weight)
        return loss


class CustomSSIMLoss(nn.Module):

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.ssim = SSIM(
            data_range=1,
            nonnegative_ssim=True,
        )
        self.weight: float = weight
        self.mse = nn.MSELoss()

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:

        pixel_loss = self.mse(X, Y) * 10  # pixel loss
        ssim_loss = 1 - self.ssim(X, Y.to(X.dtype))  # ssim loss

        # loss = feature_loss * 0.5 + pixel_loss * 0.5
        loss = ssim_loss * self.weight + pixel_loss * (1 - self.weight)
        return loss


class MS_SSIMLoss(MS_SSIM):

    def forward(self, img1, img2):
        return 100 * (1 - super(MS_SSIMLoss, self).forward(img1, img2))


class SSIMLoss(SSIM):

    def forward(self, img1, img2):
        return 100 * (1 - super(SSIMLoss, self).forward(img1, img2))


TEST_IMAGE_PATH = "./datasets/DIV2K/DIV2K_valid_HR/0801.png"
TEST_IMAGES_PATHS = [
    "./datasets/DIV2K/DIV2K_valid_HR/0801.png",
    "./datasets/DIV2K/DIV2K_valid_HR/0802.png",
    "./datasets/DIV2K/DIV2K_valid_HR/0803.png",
    "./datasets/DIV2K/DIV2K_valid_HR/0804.png",
]


def test_loss():

    for path in TEST_IMAGES_PATHS:
        original = Image.open(path)
        preprocessor = preprocessing.Unsharpen(patch_size=1024)
        sample, target = preprocessor(original)

        sample_numpy = np.array(sample[:3, ...].permute(1, 2, 0))
        target_numpy = np.array(target[:3, ...].permute(1, 2, 0))

        plt.figure(figsize=(40, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(original)  # type: ignore
        plt.subplot(1, 3, 2)
        plt.imshow(sample_numpy)  # type: ignore
        plt.subplot(1, 3, 3)
        plt.imshow(target_numpy)  # type: ignore
        plt.show()

        sample, target = sample[:3, ...].unsqueeze(0), target.unsqueeze(0)

        ssim = CustomSSIMLoss(weight=1)(sample, target)
        perceptual = PerceptualLoss(weight=1)(sample, target)
        mse = CustomSSIMLoss(weight=0)(sample, target)
        mse2 = PerceptualLoss(weight=0)(sample, target)
        print(f"{ssim=}, {perceptual=}, {mse=}, {mse2=}")

        print(
            f"perfect match: ssim = {CustomSSIMLoss(weight=1)(sample, sample)}")
