# model based on EDSR
import lightning as L
import torch
from torch import nn, Tensor
from typing import Any, Tuple, List, Optional


class PaddedConv3x3(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        if out_channels is None:
            out_channels = in_channels
        super().__init__(in_channels, out_channels, kernel_size=3, padding=1)


class MeanShift(nn.Conv2d):

    def __init__(self,
                 rgb_range: int = 255,
                 rgb_mean: Tuple[float, float,
                                 float] = (0.4488, 0.4371, 0.4040),
                 rgb_std: Tuple[float, float, float] = (1., 1., 1.),
                 sign: int = -1):

        super().__init__(3, 3, kernel_size=1)
        std = Tensor(rgb_std)
        mean = Tensor(rgb_mean)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * mean / std  # type: ignore
        for p in self.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    """
        Residual block component of EDSR. Consists of:
            - Convolution
            - ReLU
            - Convolution
        Note: no batch normalization
    """

    def __init__(self, n_features: int, residual_scaling: float = 1.0):
        super().__init__()
        self.residual_scaling: float = residual_scaling
        self.body: nn.Sequential = nn.Sequential(
            PaddedConv3x3(n_features),
            nn.ReLU(),
            PaddedConv3x3(n_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        res: Tensor = self.body(x).mul(self.residual_scaling)
        res += x
        return res


class Upsampler2X(nn.Sequential):
    """
        Upsamples the input by a factor of 2 applying a convolution
        and pixel shuffle.
    """

    def __init__(self, n_features: int):
        super().__init__(
            PaddedConv3x3(n_features, 1 * n_features),
            nn.PixelShuffle(upscale_factor=1),
        )


class EDSR(L.LightningModule):
    uses_sigmoid: bool = False

    def __init__(self,
                 n_features: int = 256,
                 residual_scaling: float = 0.1,
                 n_resblocks: int = 32,
                 loss_fn: nn.Module = nn.L1Loss(),
                 learning_rate: float = 1e-4):

        super().__init__()
        # hyperparameters
        self.n_resblocks: int = n_resblocks
        self.n_features: int = n_features
        self.residual_scaling: float = residual_scaling
        self.loss_fn: nn.Module = loss_fn
        self.lr: float = learning_rate
        self.save_hyperparameters(ignore=('loss_fn',))

        # architecture
        self.head: nn.Conv2d = PaddedConv3x3(3, self.n_features)
        self.sub_mean: MeanShift = MeanShift()
        self.add_mean: MeanShift = MeanShift(sign=1)

        # main body: a series of residual blocks plus a final conv
        self.body: nn.Sequential = nn.Sequential(
            *(ResidualBlock(self.n_features, self.residual_scaling)
              for _ in range(self.n_resblocks)),
            PaddedConv3x3(self.n_features),
        )

        # tail wihtout upsampling: no super resolution, just sharpening
        self.tail: nn.Sequential = nn.Sequential(
            Upsampler2X(self.n_features),
            PaddedConv3x3(self.n_features, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def _shared_step(self, batch: List[torch.Tensor], prefix: str) -> Tensor:
        lowres, highres = batch
        prediction = self(lowres)
        loss = self.loss_fn(prediction, highres)
        self.log(f'{prefix}_loss', loss, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, 'valid')

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(),
                                lr=self.lr,
                                betas=(0.9, 0.999),
                                eps=1e-8)
