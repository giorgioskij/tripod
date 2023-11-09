import lightning as L
from typing import Optional, List, Tuple
import torch
from torch import Tensor, nn
from torchvision import models
from torchvision.transforms import functional as F
from torchinfo import summary
from unet import DoubleConv


# Custom U-Net with ResNet34 (pretrained on Imagenet) as an encoder
class Kolnet(L.LightningModule):

    def __init__(
            self,
            loss_fn: nn.Module = nn.L1Loss(),
            learning_rate: float = 1e-3,
            freeze_encoder: bool = True,
            use_espcn: bool = False,
            use_espcn_activations: bool = True,
            avoid_deconv: bool = False,
            use_alpha: bool = False,
            double_image_size: bool = False,
            metrics: nn.ModuleDict = nn.ModuleDict(),
    ):
        super().__init__()
        # hyperparams
        self.lr: float = learning_rate
        # self.last_activation: Optional[nn.Module] = last_activation
        self.use_espcn_activations: bool = use_espcn_activations
        self.loss_fn: nn.Module = loss_fn
        self.freeze_encoder: bool = freeze_encoder
        self.use_espcn: bool = use_espcn
        self.avoid_deconv: bool = avoid_deconv
        self.use_alpha: bool = use_alpha
        self.double_image_size: bool = double_image_size
        self.metrics: nn.ModuleDict = metrics

        # encoder - pretrained resnet
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if self.use_alpha:
            self.conv1 = nn.Conv2d(4,
                                   64,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
        else:
            self.conv1 = self.encoder.conv1

        # decoder
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.double_conv_512_512 = DoubleConv(512, 512)
        self.double_conv_1024_512_256 = DoubleConv(1024, 256, mid_channels=512)
        self.double_conv_512_256_128 = DoubleConv(512, 128, mid_channels=256)
        self.double_conv_256_128_64 = DoubleConv(256, 64, mid_channels=128)
        self.double_conv_128_64_64 = DoubleConv(128, 64, mid_channels=64)
        self.double_conv_64_64_nobn = DoubleConv(64, 64, use_bn=False)
        self.double_conv_64_32_3_nobn = DoubleConv(64, 3, 32, use_bn=False)

        self.upscale = nn.Upsample(scale_factor=2,
                                   mode="bilinear",
                                   align_corners=True)

        self.deconv_64_3 = nn.ConvTranspose2d(
            64,
            3,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        # if self.use_espcn:
        self.conv5x5_6_64 = nn.Conv2d(6, 64, kernel_size=5, padding=2)
        self.conv3x3_64_32 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3x3_32_12 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # else:
        self.deconv_6_3 = nn.ConvTranspose2d(6,
                                             3,
                                             kernel_size=7,
                                             stride=2,
                                             padding=3,
                                             bias=False)

        self.relu = nn.ReLU(inplace=True)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=["metrics"])

    def forward(self, x: Tensor) -> Tensor:
        # all comments refer to an example input of size (3, 64, 64)

        # x64 = self.encoder.conv1(x)  # (64, 32, 32)
        x64 = self.conv1(x)  # (64, 32, 32)
        x64 = self.encoder.bn1(x64)
        x64 = self.encoder.relu(x64)
        x64 = self.encoder.maxpool(x64)  # (64, 16, 16)
        x64 = self.encoder.layer1(x64)
        x128 = self.encoder.layer2(x64)  # (128, 8, 8)
        x256 = self.encoder.layer3(x128)  # (256, 4, 4)
        x512 = self.encoder.layer4(x256)  # (512, 2, 2)

        x512_up = self.double_conv_512_512(x512)  # (512, 2, 2)

        # bottleneck: 1024 -> 256
        # x512_up = self.upscale(x512_up) # (512, 4, 4)
        x1024_up = torch.cat((x512_up, x512), dim=-3)  # 2(512, 2x2)=(1024, 2x2)
        x256_up = self.double_conv_1024_512_256(x1024_up)  # (256, 2, 2)

        # upscale 1: 512 -> 128
        x256_up = self.upscale(x256_up)  # (256, 4, 4)
        x512_up = torch.cat((x256_up, x256), dim=-3)  # 2(256, 4x4)=(512, 4x4)
        x128_up = self.double_conv_512_256_128(x512_up)  # (128, 4, 4)

        # upscale 2: 256 -> 64
        x128_up = self.upscale(x128_up)  # (128, 8, 8)
        x256_up = torch.cat((x128_up, x128), dim=-3)  # 2(128, 8x8)=(256, 8x8)
        x64_up = self.double_conv_256_128_64(x256_up)  # (64, 8, 8)

        # upscale 3: 128 -> 64
        x64_up = self.upscale(x64_up)  # (64, 16, 16)
        x128_up = torch.cat((x64_up, x64), dim=-3)  # 2(64, 16x16)=(128, 16x16)
        x64_up = self.double_conv_128_64_64(x128_up)  # (64, 16, 16)

        # upscale 4:
        x64_up = self.upscale(x64_up)  # (64, 32, 32)
        x64_up = self.double_conv_64_64_nobn(x64_up)  # (64, 32, 32)

        # upscale 5 if applying superresolution
        if self.avoid_deconv:
            x3_up = self.upscale(x64_up)  # (64, 64, 64)
            x3_up = self.double_conv_64_32_3_nobn(x3_up)  # (3, 64, 64)

        else:
            x3_up = self.deconv_64_3(
                x64_up,
                output_size=(x64_up.shape[-2] * 2, x64_up.shape[-1] * 2),
            )  # (3, 64, 64)

        # concatenate decoded (3, 64x64) and input (3, 64x64). (If input has 4
        # channels, ignore the 4th)
        x6_up = torch.cat((x3_up, x[:, :3, :, :]), dim=-3)  # (6, 64, 64)

        if self.use_espcn:
            # first conv
            output = self.conv5x5_6_64(x6_up)  # (64, 64, 64)
            # activation
            output = (self.tanh(output)
                      if self.use_espcn_activations else self.relu(output))
            # second conv
            output = self.conv3x3_64_32(output)  # (32, 64, 64)
            # activation
            output = (self.tanh(output)
                      if self.use_espcn_activations else self.relu(output))
            # third conv
            output = self.conv3x3_32_12(output)  # (12, 64, 64)
            # pixel shuffle
            output = self.pixel_shuffle(output)  # (3, 128, 128)
            if self.use_espcn_activations:
                output = self.sigmoid(output)

        else:
            # finally, apply super resolution: upscale (6, 64x64) -> (3, 128x128)
            output = self.deconv_6_3(
                x6_up,
                output_size=(x6_up.shape[-2] * 2, x6_up.shape[-1] * 2),
            )  # (3, 128, 128)

        if not self.double_image_size:
            output = F.resize(output,
                              [output.size(-1) // 2,
                               output.size(-2) // 2],
                              antialias=True)

        if output.isnan().any():
            raise ArithmeticError("Detected nan values in kolnet forward")

        return output

    def _shared_step(self, batch: List[Tensor], prefix: str):
        x, y = batch
        prediction = self(x)
        y = y.to(prediction.dtype)
        loss = self.loss_fn(prediction, y)
        if isinstance(loss, Tuple):
            loss, feature_loss, pixel_loss = loss
            self.log(f"{prefix}/_perceptual_loss", feature_loss, prog_bar=False)
            self.log(f"{prefix}/_pixel_loss", pixel_loss, prog_bar=False)

        self.log(f"{prefix}/_loss", loss, prog_bar=True)

        for metric_name, metric_fn in self.metrics.items():
            with torch.no_grad():
                metric_value = metric_fn(prediction, y)
            self.log(f"{prefix}/_{metric_name}", metric_value, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "valid")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def sharpen(self, image: Tensor, amount: float):
        if len(image.shape) != 3 or image.shape[0] != 3:
            raise ValueError("Give image as a single tensor of shape [3, W, H]"
                             f"\nReceived shape was {image.shape}")

        alpha = torch.ones(image.shape[1:]).unsqueeze(0) * amount
        image = torch.cat((image, alpha), dim=0)

        image = image.unsqueeze(0).to(list(self.parameters())[0].device)

        return self(image)


if __name__ == "__main__":
    n = Kolnet(
        loss_fn=nn.MSELoss(),
        learning_rate=1e-3,
        freeze_encoder=True,
        use_espcn=True,
        use_espcn_activations=True,
        avoid_deconv=True,
        use_alpha=True,
        double_image_size=False,
    )

    summary(n, (8, 4, 512, 512))
