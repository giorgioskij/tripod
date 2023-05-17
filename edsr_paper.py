import common_paper
from dataclasses import dataclass
import torch.nn as nn
import torch
from torch import Tensor
from typing import List, Any
import lightning as L

url = {
    'r16f64x2':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4':
        'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


def make_model(args, parent=False):
    return EDSR(args)


@dataclass
class Args:
    rgb_range: int = 255
    n_resblocks: int = 16
    n_feats: int = 64
    scale: int = 2
    res_scale: float = 1
    patch_size: int = 96
    n_colors: int = 3


class EDSR(nn.Module):

    def __init__(self, args, conv=common_paper.default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common_paper.MeanShift(args.rgb_range)
        self.add_mean = common_paper.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body: List[nn.Module] = [
            common_paper.ResBlock(conv,
                                  n_feats,
                                  kernel_size,
                                  act=act,
                                  res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common_paper.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, '
                            'whose dimensions in the model are {} and '
                            'whose dimensions in the checkpoint are {}.'.format(
                                name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(
                        'unexpected key "{}" in state_dict'.format(name))


class EDSRLightning(L.LightningModule):

    uses_sigmoid: bool = False

    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.model = EDSR(Args())
        self.lr: float = learning_rate
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: List[torch.Tensor], prefix: str) -> Tensor:
        lowres, highres = batch
        prediction = self(lowres)
        loss = nn.L1Loss()(prediction, highres)
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
