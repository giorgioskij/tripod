from pathlib import Path

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms as T
from torchvision.datasets import MNIST, Flowers102
import torchvision
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from torch import Tensor
import os


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


class SuperDataModule(L.LightningDataModule):

    def __init__(
        self,
        dataset: str = "MNIST",
        data_dir: Path = Path("./data"),
        batch_size_train: int = 32,
        batch_size_test: int = 256,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.dataset = dataset

    # def prepare_data(self) -> None:
    # download if not downloaded
    # if self.dataset == "MNIST":
    #     CustomMNIST("./data", download=True, train=True)
    #     CustomMNIST("./data", download=True, train=False)
    # elif self.dataset.lower() == "flowers":

    def setup(self, stage=None) -> None:
        self.prepare_data()

        if self.dataset == "MNIST":
            sample_transform = T.Compose(
                [T.GaussianBlur(7, sigma=(1, 2)),
                 T.ToTensor()])
            target_transform = T.ToTensor()
            self.train = CustomMNIST(
                "./data",
                train=True,
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.test = CustomMNIST(
                "./data",
                train=False,
                transform=sample_transform,
                target_transform=target_transform,
            )

        elif self.dataset.lower() == "flowers":
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
                "./data",
                download=True,
                split="train",
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.val = CustomFlowers(
                "./data",
                download=True,
                split="val",
                transform=sample_transform,
                target_transform=target_transform,
            )
            self.test = CustomFlowers(
                "./data",
                download=True,
                split="test",
                transform=sample_transform,
                target_transform=target_transform,
            )

        # loaders for demo
        self.trainset_iter = iter(self.train)
        self.trainloader_iter = iter(self.train_dataloader(4))
        self.testset_iter = iter(self.test)
        self.testloader_iter = iter(self.test_dataloader(4))

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_train
        return torch.utils.data.DataLoader(self.train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val,
                                           batch_size=self.batch_size_test,
                                           shuffle=False,
                                           num_workers=0)

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size_test
        return torch.utils.data.DataLoader(self.test,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)

    def demo_image(self, train=True):
        if train:
            return next(self.trainset_iter)
        else:
            return next(self.testset_iter)

    def demo_batch(self, train=True) -> List[Tensor]:
        if train:
            return next(self.trainloader_iter)
        else:
            return next(iter(self.testloader_iter))


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
    return t.detach().cpu().permute(1, 2, 0).numpy()


def show(b: Tuple | List | Tensor, save_path: Optional[Path] = None) -> None:
   
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
    elif isinstance(b, Tensor) and len(b.shape) == 3:
        im = tensor_to_image(b)
        if save_path is not None:
            if not save_path.exists():
                save_path.mkdir(parents = True)
            plt.imsave(str(save_path / "output.png"), im)
        plt.imshow(im, cmap="gray")
        plt.show()

    # batch of images
    elif isinstance(b, Tensor) and len(b.shape) == 4:
        f, ax = plt.subplots(1, len(b))
        for i, img in enumerate(b):
            img = tensor_to_image(img)
            if save_path is not None:
                if not save_path.exists():
                    save_path.mkdir(parents=True)
                plt.imsave(str(save_path / f"output{i}.png"), img)
            ax[i].imshow(img, cmap="gray")
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
