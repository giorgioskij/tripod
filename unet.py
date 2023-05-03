from typing import Any
import lightning as L
import torch
from torch import Tensor
from torch import nn
from typing import *
import torchvision.transforms.functional


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        # upscale x with transpose convolution
        x = self.upconv(x)
        # crop residual in the center to the size of x
        residual = torchvision.transforms.functional.center_crop(
            residual, list(x.shape[-2:]))
        # concatenate residual and x along channels
        concatenation = torch.cat((residual, x), dim=-3)
        # apply double convolution
        concatenation = self.double_conv(concatenation)
        return concatenation


class Downscale(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x


class UNet(L.LightningModule):

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # encoder
        self.input_conv = DoubleConv(in_channels, 64)
        self.downscale1 = Downscale(64, 128)
        self.downscale2 = Downscale(128, 256)
        self.downscale3 = Downscale(256, 512)
        self.downscale4 = Downscale(512, 1024)

        # decoder
        self.upscale1 = Upscale(1024, 512)
        self.upscale2 = Upscale(512, 256)
        self.upscale3 = Upscale(256, 128)
        self.upscale4 = Upscale(128, 64)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        # encode
        x1 = self.input_conv(x)
        x2 = self.downscale1(x1)
        x3 = self.downscale2(x2)
        x4 = self.downscale3(x3)
        x = self.downscale4(x4)

        # decode
        x = self.upscale1(x, x4)
        x = self.upscale2(x, x3)
        x = self.upscale3(x, x2)
        x = self.upscale4(x, x1)
        x = self.output_conv(x)
        return x

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        prediction = torch.sigmoid(logits)

        loss = self.loss_fn(prediction, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        prediction = torch.sigmoid(logits)
        loss = self.loss_fn(prediction, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
