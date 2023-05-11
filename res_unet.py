import segmentation_models_pytorch as smp
import lightning as L
from typing import Callable, List
from torch import Tensor
import torch


# a lightning module that uses u as its model
class UResNet(L.LightningModule):

    def __init__(
        self,
        loss_fn: Callable,
        lr: float = 1e-3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=3)
        self.lr = lr
        self.loss_fn: Callable = loss_fn

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: List[Tensor], prefix: str):
        x, y = batch
        logits = self(x)
        prediction = torch.sigmoid(logits)
        # prediction = torch.clip(logits, 0, 1)
        loss = self.loss_fn(prediction, y.to(prediction.dtype))
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
