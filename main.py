from data import SuperDataModule, show
from cnn import Cnn
import lightning as L
from lightning.pytorch import loggers
from matplotlib import pyplot as plt
from unet import PoolingStrategy, UNet
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from res_unet import UResNet
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
torch.set_float32_matmul_precision("medium")

# load data and model
d = SuperDataModule(batch_size_train=4, batch_size_test=16, dataset="flowers")
d.setup()

b = d.demo_batch()
show(b)
u = UResNet(loss_fn=torch.nn.MSELoss(), pretrained=True)

trainer = L.Trainer(
    accelerator="auto",
    max_epochs=3,
    logger=loggers.CSVLogger(save_dir="./", version=2),
    precision="16-mixed",
)

# u = UNet.load_from_checkpoint(
#     "./lightning_logs/version_0/checkpoints/1epoch.ckpt").cpu()
# u = UNet(loss_fn=torch.nn.MSELoss(), pooling_strategy=PoolingStrategy.conv)
# trainer = L.Trainer(
#     accelerator="auto",
#     max_epochs=3,
#     logger=loggers.CSVLogger(save_dir="./", version=1),
#     precision="16-mixed",
# )

# train
trainer.fit(model=u, datamodule=d)
trainer.test(model=u, datamodule=d)

# demo
test_batch = d.demo_batch(train=True)
output = u(test_batch[0].to(u.device))
output = torch.sigmoid(output)

# p = Path("./") / "outputs/version1/"
# p.mkdir(parents=True)

output_path = Path("./") / "outputs/version2/batch1"
show(test_batch, output_path)
show(output, output_path)


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
