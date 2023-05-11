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
from pytorch_msssim import SSIM
from loss import TripodLoss

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*full_state_update.*")
torch.set_float32_matmul_precision("medium")

# load data and model
d: SuperDataModule = SuperDataModule(batch_size_train=4,
                                     batch_size_test=16,
                                     dataset="flowers")
d.setup()

b = d.demo_batch()
show(b)

# ResNet encoder
# u = UResNet(
#     # loss_fn=SSIM(data_range=1, size_average=True, nonnegative_ssim=True),
#     # loss_fn=torch.nn.MSELoss(),
#     loss_fn=TripodLoss(),
#     pretrained=True,
# )
# trainer = L.Trainer(
#     accelerator="auto",
#     max_epochs=3,
#     logger=loggers.CSVLogger(save_dir="./", version=2),
#     precision="16-mixed",
# )

# classic encoder
u = UNet(loss_fn=TripodLoss(),
         pooling_strategy=PoolingStrategy.conv,
         bilinear_upsampling=True)
trainer = L.Trainer(
    accelerator="auto",
    max_epochs=3,
    logger=loggers.CSVLogger(save_dir="./", version=3),
    precision="16-mixed",
)

# u = UNet.load_from_checkpoint(
#     "./lightning_logs/version_0/checkpoints/1epoch.ckpt").cpu()

# train
trainer.fit(model=u, datamodule=d)
trainer.test(model=u, datamodule=d)

# demo
test_batch = d.demo_batch(train=False)
output = u(test_batch[0].to(u.device))
output = torch.sigmoid(output)
show(test_batch)
show(output)

# save output
output_path = Path("./") / "outputs/bilinear_tripodloss/batch1"
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
