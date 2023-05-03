from data import BlurredMNIST, show
from cnn import Cnn
import lightning as L
from lightning.pytorch import loggers
from matplotlib import pyplot as plt
from unet import UNet
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch

# load data and model
d = BlurredMNIST(batch_size_train=4, batch_size_test=16, dataset="flowers")
d.setup()

b = d.demo_batch()
show(b)

# u = UNet.load_from_checkpoint(
#     "./lightning_logs/version_0/checkpoints/1epoch.ckpt").cpu()

# train
u = UNet(in_channels=3, out_channels=3)
trainer = L.Trainer(
    accelerator="mps",
    max_epochs=1,
    # max_steps=100,
    logger=loggers.CSVLogger(save_dir="./", version=0),
    # log_every_n_steps=1,
)
trainer.fit(model=u, datamodule=d)
trainer.test(model=u, datamodule=d)

# demo
test_batch = d.demo_batch(train=True)
output = u(test_batch[0].to(u.device))
output = torch.sigmoid(output)

show(test_batch)
show(output)


def check_worst_case():
    worst_loss = 0
    u.eval()
    worst_blurred, worst_original, worst_prediction = None, None, None
    for blurred, original in d.test_dataloader():
        logits = u(blurred.to(u.device))
        prediction = torch.sigmoid(logits)
        loss = u.loss_fn(prediction, original.to(u.device))
        if loss > worst_loss:
            worst_loss = loss
            worst_blurred = blurred
            worst_original = original
            worst_prediction = prediction
