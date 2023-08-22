"""
Handles everything that happens to the images before being fed to the neural
network, including creation of training images and data augmentation.
"""

from PIL import Image
from typing import Tuple, Optional
from torch import Tensor
from torchvision import transforms as T
import albumentations as A
import numpy as np
import torch
import random
import math
from math import pi as PI
from matplotlib import pyplot as plt

from data import show

TEST_IMAGE_PATH = "./datasets/DIV2K/DIV2K_valid_HR/0801.png"


def unsharpen(sample: Image.Image,
              amount: Optional[float] = None,
              patch_size: int = 128) -> Tuple[Tensor, Tensor]:

    if amount is None:
        amount = random.random()
    if amount < 0 or amount > 1:
        raise ValueError("amount has to be between 0 and 1")

    # random crop
    sample = T.RandomCrop(patch_size)(sample)
    target = sample.copy()

    # kernel size from 0% to 10% of the image size, based on amount parameter
    kernel_size = int(round(patch_size * 0.1 * amount))

    # motion blur
    if kernel_size > 2:
        sample = MotionBlur(blur_limit=(kernel_size, kernel_size),
                            always_apply=True)(image=np.array(sample))["image"]

    to_tensor = T.ToTensor()

    sample_tensor = to_tensor(sample)
    target_tensor = to_tensor(target)
    # add amount value to alpha channel of image
    sample_tensor = torch.cat(
        (sample_tensor, torch.ones(1, *sample_tensor.shape[-2:]) * amount),
        dim=0)
    return sample_tensor, target_tensor


def test_albumentation(transform, crop=True):
    im = Image.open(TEST_IMAGE_PATH)
    if crop:
        im = T.RandomCrop(128)(im)

    plt.imshow(im)  #type: ignore
    plt.show()
    plt.imshow(transform(image=np.array(im))["image"])
    plt.show()


def tripod_transforms(sample: Image.Image) -> Tuple[Tensor, Tensor]:
    # randomly crop 128
    common_transforms = T.Compose([
        T.RandomCrop(128),
    ])
    crop = T.RandomCrop(128),
    sample = common_transforms(sample)
    target = sample.copy()

    # downscale sample
    sample = T.Resize(64)(sample)

    # albumentations
    sample_transforms = A.Compose([
        # A.Downscale(interpolation=cv2.INTER_LINEAR),
        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.5)),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 3)),
        # albumentations.pytorch.transforms.ToTensorV2(),
    ])
    sample = sample_transforms(image=np.array(sample))["image"]

    # convert both to tensors
    totensor = T.ToTensor()
    sample_tensor = totensor(sample)
    target_tensor = totensor(target)

    return sample_tensor, target_tensor


from albumentations import Blur
from typing import Union, Dict, Any
from albumentations.augmentations import functional as FMain
import cv2

ScaleIntType = Union[int, Tuple[int, int]]


class MotionBlur(Blur):
    """Modification to the Albumentation class MotionBlur. Here, the length of
    the blur line is fixed so that controlling the kernel size via the 
    blur_limit parameter directly controls the intensity of the effect, instead
    of just giving an upper bound to it. 

    Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        blur_limit: ScaleIntType = 7,
        # allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return super().get_transform_init_args_names() + ("allow_shifted",)

    def apply(self,
              img: np.ndarray,
              kernel: Optional[np.ndarray] = None,
              **params) -> np.ndarray:  # type: ignore
        return FMain.convolve(img, kernel=kernel)

    def get_params(self) -> Dict[str, Any]:
        ksize = random.choice(
            list(
                range(
                    self.blur_limit[0],  #type: ignore
                    self.blur_limit[1] + 1,  #type: ignore
                    2)))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)

        # modification: instead of picking two random points and making a line
        # between them, pick a direction and make a line of fixed length in
        # that direction
        angle_x1 = 2 * PI * random.random()
        angle_x2 = angle_x1 + PI if angle_x1 < PI else angle_x1 - PI
        radius = ksize // 2
        center = ksize / 2
        x1 = int(round(math.cos(angle_x1) * radius + center))
        x2 = int(round(math.cos(angle_x2) * radius + center))
        y1 = int(round(math.sin(angle_x1) * radius + center))
        y2 = int(round(math.sin(angle_x2) * radius + center))

        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)  # type: ignore

        # Normalize kernel
        return {"kernel": kernel.astype(np.float32) / np.sum(kernel)}