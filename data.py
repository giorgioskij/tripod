from enum import Enum, auto
from pathlib import Path
from random import sample
from typing import Any, Callable, List, Optional, Tuple

import lightning as L
import torch.utils.data
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms as T
from torchvision.datasets import MNIST, Flowers102
import albumentations as A
import cv2
import math
import numpy as np

from div2k import DIV2K
from gopro import GoProTrainDataset, GoProValDataset

plt.axis('off')


class Dataset(Enum):
    DIV2K = auto()
    MNIST = auto()
    FLOWERS = auto()
    GOPRO = auto()


class CustomMNIST(MNIST):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        # img, target = self.data[index], int(self.targets[index])
        img = self.data[index]
        # throw away the label - use as label the original image
        img = Image.fromarray(img.numpy(), mode="L")
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CustomFlowers(Flowers102):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = Image.open(image_file).convert("RGB")
        target = image.copy()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            # label = self.target_transform(label)
            target = self.target_transform(target)

        return image, target


class TripodDataModule(L.LightningDataModule):

    def __init__(
        self,
        dataset: Dataset = Dataset.DIV2K,
        data_dir: Path = Path("./datasets"),
        patch_size: Optional[int] = None,
        # batch_size: Optional[int] = None,
        batch_size_train: int = 16,
        batch_size_test: int = 64,
        sample_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        common_transform: Optional[Callable] = None,
        sample_target_generator: Optional[Callable[[Any],
                                                   Tuple[Tensor,
                                                         Tensor]]] = None,
    ):
        """TripodDataModule for sharpening

        Args:
            dataset (Dataset, optional): Defaults to Dataset.FLOWERS.
            data_dir (Path, optional): Defaults to Path("./datasets").
            batch_size (Optional[int], optional): Defaults to None.
            batch_size_train (int, optional): Defaults to 32.
            batch_size_test (int, optional): Defaults to 256.
        """

        super().__init__()
        self.patch_size: Optional[int] = patch_size
        self.data_dir: Path = data_dir
        self.batch_size_train: int = batch_size_train
        self.batch_size_test: int = batch_size_test
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.common_tranform = common_transform
        self.sample_target_generator: Optional[Callable[[Any], Tuple[
            Tensor, Tensor]]] = sample_target_generator

        # if batch_size is not None:
        #     self.batch_size = batch_size
        #     self.batch_size_train = batch_size
        #     self.batch_size_test = batch_size
        self.dataset: Dataset = dataset

    # def prepare_data(self) -> None:
    # download if not downloaded
    # if self.dataset == "MNIST":
    #     CustomMNIST("./data", download=True, train=True)
    #     CustomMNIST("./data", download=True, train=False)
    # elif self.dataset.lower() == "flowers":

    def setup(self, stage=None) -> None:
        self.prepare_data()

        if self.dataset == Dataset.MNIST:
            raise NotImplementedError(
                "Implementation not up to date. Use DIV2K.")
            sample_transform = T.Compose(
                [T.GaussianBlur(7, sigma=(1, 2)),
                 T.ToTensor()])
            target_transform = T.ToTensor()
            self.train = CustomMNIST(
                self.data_dir,
                train=True,
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.val = CustomMNIST(
                self.data_dir,
                train=False,
                transform=sample_transform,
                target_transform=target_transform,
            )

        elif self.dataset == Dataset.FLOWERS:
            raise NotImplementedError(
                "Implementation not up to date. Use DIV2K.")
            sample_transform = T.Compose([
                T.Resize((416, 416)),
                T.GaussianBlur(9, sigma=(1, 5)),
                T.ToTensor(),
            ])
            target_transform = T.Compose([
                T.Resize((416, 416)),
                T.ToTensor(),
            ])
            self.train = CustomFlowers(
                self.data_dir,
                download=True,
                split="train",
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.val = CustomFlowers(
                self.data_dir,
                download=True,
                split="val",
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.test = CustomFlowers(
                self.data_dir,
                download=True,
                split="test",
                transform=sample_transform,
                target_transform=target_transform,
            )

        elif self.dataset == Dataset.DIV2K:

            self.train = DIV2K(
                root_dir=self.data_dir,
                train=True,
                sample_target_generator=self.sample_target_generator,
                download=True)
            self.val = DIV2K(
                root_dir=self.data_dir,
                train=False,
                sample_target_generator=self.sample_target_generator,
                download=True)

        elif self.dataset == Dataset.GOPRO:
            self.train = GoProTrainDataset(self.data_dir / "GoPro" / "train",
                                           patch_size=self.patch_size or 256)
            self.val = GoProValDataset(self.data_dir / "GoPro" / "test",
                                       patch_size=self.patch_size or 256)

        # loaders for demo
        self.trainset_iter = iter(self.train)
        self.trainloader_iter = iter(self.train_dataloader(4))
        self.valset_iter = iter(self.val)
        self.valloader_iter = iter(self.val_dataloader(4))

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_train
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=10)

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_test
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=10)

    def test_dataloader(self, batch_size=None):
        raise NotImplementedError()
        # if batch_size is None:
        #     batch_size = self.batch_size_test
        # return torch.utils.data.DataLoader(self.test,
        #                                    batch_size=batch_size,
        #                                    shuffle=False,
        #                                    num_workers=10)

    def demo_image(self, train=True):
        if train:
            return next(self.trainset_iter)
        else:
            return next(self.valset_iter)

    def demo_batch(self, train=True) -> List[Tensor]:
        if train:
            return next(self.trainloader_iter)
        else:
            return next(self.valloader_iter)


def display_batch(b, predictions=None, max_images=4):
    images, labels = b
    n_images = min(len(images), max_images)
    f, axarr = plt.subplots(1, n_images)
    for i, image in enumerate(images):
        if i >= n_images:
            break
        npimage = image.detach().cpu().permute(1, 2, 0).numpy()
        labelstring = f"label: {labels[i]}"
        if predictions is not None:
            labelstring += f"\nprediction: {predictions[i]}"
        axarr[i].set_title(labelstring)
        axarr[i].imshow(npimage, cmap="gray")
    plt.show()


def tensor_to_image(t: Tensor):
    return t.detach().cpu().permute(1, 2, 0).clip(0, 1).numpy()


def show(b: Tuple | List | Tensor,
         save_path: Optional[Path] = None,
         ignore_alpha: bool = True,
         title: Optional[str | List] = None) -> None:

    # two batches of images
    if (isinstance(b, tuple) or
            isinstance(b, list)) and len(b) == 2 and isinstance(
                b[0], Tensor) and isinstance(b[1], Tensor):

        if save_path is not None:
            show(b[0], save_path / "transformed")
            show(b[1], save_path / "original")
        else:
            show(b[0])
            show(b[1])

    # single image
    elif isinstance(b, Tensor) and (len(b.shape) == 3 or
                                    len(b.shape) == 4 and b.shape[0] == 1):
        b = b.squeeze()
        if ignore_alpha and b.shape[0] == 4:
            b = b[:3, :, :]
        im = tensor_to_image(b)
        if save_path is not None:
            if not save_path.exists():
                save_path.mkdir(parents=True)
            plt.imsave(str(save_path / "img.png"), im)
        # plt.imshow(im, cmap="gray")
        plt.imshow(im)
        plt.show()

    # batch of images
    elif isinstance(b, Tensor) and len(b.shape) == 4:
        if ignore_alpha and b.shape[1] == 4:
            b = b[:, :3, :, :]

        # n_cols = 1 if len(b) <= 5 else 5 if len(b) % 5 == 0 else 4
        n_cols = 2
        n_rows = int(math.ceil(len(b) / n_cols))
        dpi = 100
        imwidth = b.shape[-1] * n_cols + b.shape[-1] // 10 * n_cols
        imheight = b.shape[-2] * n_rows + b.shape[-2] // 10 * n_rows
        f, ax = plt.subplots(n_rows,
                             n_cols,
                             figsize=(imwidth // dpi, imheight // dpi),
                             dpi=dpi)

        for i, img in enumerate(b):
            curr_ax = ax[i // n_cols][i % n_cols]
            curr_ax.xaxis.set_visible(False)
            curr_ax.yaxis.set_visible(False)
            img = tensor_to_image(img)
            if save_path is not None:
                if not save_path.exists():
                    save_path.mkdir(parents=True)
                plt.imsave(str(save_path / f"img{i}.png"), img)
            # ax[i].imshow(img, cmap="gray")
            if title is not None and isinstance(title, List):
                curr_ax.set_title(title[i])
            curr_ax.imshow(img)
        plt.show()

    else:
        raise ValueError("Invalid input")

    return


# flowers_test = torchvision.datasets.Flowers102("./data",
#                                                download=True,
#                                                split="val")

# dl = torch.utils.data.DataLoader(flowers_train, batch_size=4)

# it = iter(flowers_train)
# show(next(it)[0])
# it = iter(dl)
# b = next(it)
# show(b)
if __name__ == "__main__":
    ...
    # d = TripodDataModule(dataset=Dataset.GOPRO)
    # d.setup()
    # dl = d.train_dataloader()

    # print(next(iter(dl)))
