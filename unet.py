from typing import Any
import lightning as L
import torch
from torch import Tensor
from torch import nn
from typing import Callable, List
import torchvision.transforms.functional
from torchinfo import summary
from torchvision import models
import segmentation_models_pytorch as smp
from enum import Enum, auto

# res = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
# summary(res)
# u = smp.Unet(encoder_name="resnet34",
#              encoder_weights=None,
#              in_channels=3,
#              classes=3)


class PoolingStrategy(Enum):
    max = auto()
    conv = auto()
    stride = auto()


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        halven: bool = False,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=2 if halven else 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upscale(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bilinear: bool = False):
        super().__init__()
        self.bilinear: bool = bilinear

        if self.bilinear:
            # self.upconv = nn.Sequential(
            #     nn.UpsamplingBilinear2d(scale_factor=2),
            #     nn.ConvTranspose2d(in_channels,
            #                        in_channels // 2,
            #                        kernel_size=2,
            #                        stride=1),
            # )
            self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling_strategy: PoolingStrategy,
        residual: bool = False,
    ):
        super().__init__()
        self.residual: bool = residual
        self.pooling_strategy: PoolingStrategy = pooling_strategy
        halven = pooling_strategy == PoolingStrategy.stride
        self.maxpool = nn.MaxPool2d(2)
        self.downconv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=2,
                                  stride=2)
        self.res_conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1)
        self.double_conv = DoubleConv(in_channels, out_channels, halven)

    def forward(self, x: Tensor) -> Tensor:
        if self.pooling_strategy == PoolingStrategy.max:
            x = self.maxpool(x)
        elif self.pooling_strategy == PoolingStrategy.conv:
            x = self.downconv(x)

        # residual connection around double convolution
        double_conv_out = self.double_conv(x)
        if self.residual:
            res = self.res_conv(x)
            double_conv_out += res

        return double_conv_out


class UNet(L.LightningModule):
    uses_sigmoid: bool = True

    def __init__(self,
                 loss_fn: Callable,
                 pooling_strategy: PoolingStrategy,
                 bilinear_upsampling: bool = False,
                 learning_rate: float = 1e-3,
                 residual: bool = False,
                 in_channels: int = 3,
                 out_channels: int = 3):
        super().__init__()

        self.pooling_strategy: PoolingStrategy = pooling_strategy
        self.bilinear_upsampling: bool = bilinear_upsampling
        self.loss_fn: Callable = loss_fn
        self.lr: float = learning_rate
        self.residual: bool = residual

        # encoder
        self.input_conv = DoubleConv(in_channels, 64)
        self.downscale1 = Downscale(64, 128, self.pooling_strategy,
                                    self.residual)
        self.downscale2 = Downscale(128, 256, self.pooling_strategy,
                                    self.residual)
        self.downscale3 = Downscale(256, 512, self.pooling_strategy,
                                    self.residual)
        self.downscale4 = Downscale(512, 1024, self.pooling_strategy,
                                    self.residual)

        # decoder
        self.upscale1 = Upscale(1024, 512, self.bilinear_upsampling)
        self.upscale2 = Upscale(512, 256, self.bilinear_upsampling)
        self.upscale3 = Upscale(256, 128, self.bilinear_upsampling)
        self.upscale4 = Upscale(128, 64, self.bilinear_upsampling)
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.save_hyperparameters(ignore=["in_channels", "out_channels"])

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

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "train")
        return loss

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "test")
        return loss

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, "valid")
        return loss

    def _shared_step(self, batch: List[Tensor], prefix: str) -> Tensor:
        x, y = batch
        logits = self(x)
        prediction = torch.sigmoid(logits)
        loss = self.loss_fn(prediction, y)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
