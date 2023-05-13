from edsr import EDSR
from data import TripodDataModule
from torch import nn


model = EDSR(n_features=64,
             residual_scaling=1,
             n_resblocks=16,
             loss_fn=nn.L1Loss())
