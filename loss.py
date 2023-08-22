import torch
from torch import nn
from pytorch_msssim import SSIM
from torchvision import models
from torch import Tensor
from torchinfo import summary


class TripodLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # self.ssim = SSIM(data_range=1, nonnegative_ssim=True)
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

        # ssim_loss = self.ssim(X, Y.to(X.dtype))  # ssim loss

        loss = feature_loss * 0.5 + pixel_loss * 0.5
        return loss
