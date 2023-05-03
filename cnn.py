import lightning.pytorch as pl
from torch import nn
import torch
from torch import Tensor
import torchvision
import torch.utils.data
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
import torchmetrics
from typing import *


# pytorch lightning CNN to classify MNIST digits
class Cnn(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3)
        # self.adaptive_pool = nn.AdaptiveMaxPool2d((4, 4))
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(5 * 5 * 16, 10)
        self.relu = nn.ReLU()

        # metrics
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch: Tuple[Tensor, Tensor],
                      batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        predictions = logits.argmax(dim=1)
        self.log("test_loss", loss)
        self.accuracy(predictions, y)
        self.log("test_accuracy", self.accuracy)
        return loss

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        # x = self.adaptive_pool(x)
        logits = self.fc(x.flatten(1))
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
