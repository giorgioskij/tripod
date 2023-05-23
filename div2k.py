# pytorch implementation of the DIV2K dataset. Extends VisionDataset to
# integrate with torchvision

from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from typing import List, Tuple, Optional, Tuple, Callable, Any
from PIL import Image
from torch import Tensor

from torchvision.transforms import Compose


class DIV2K(VisionDataset):

    _base_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/"
    _download_url_train = _base_url + "DIV2K_train_HR.zip"
    _download_url_valid = _base_url + "DIV2K_valid_HR.zip"

    def __init__(
        self,
        root_dir: Path,
        train: bool = True,
        transform=None,
        target_transform=None,
        common_transform=None,
        sample_target_generator: Optional[Callable[[Any],
                                                   Tuple[Tensor,
                                                         Tensor]]] = None,
        download: bool = False,
    ):
        super().__init__(str(root_dir),
                         transform=transform,
                         target_transform=target_transform)
        self.train = train
        self.root_dir: Path = Path(root_dir)
        self.dataset_dir: Path = self.root_dir / "DIV2K"
        self.transform = transform
        self.target_transform = target_transform
        self.common_transform = common_transform
        self.sample_target_generator: Optional[Callable[[Any], Tuple[
            Tensor, Tensor]]] = sample_target_generator

        self.folder_basename: str = ("DIV2K_train_HR"
                                     if self.train else "DIV2K_valid_HR")

        self.image_folder = self.dataset_dir / self.folder_basename
        self.image_folder = self.image_folder.resolve()

        self.image_files: List[str] = [
            f"{i:04d}.png"
            for i in (range(1, 801) if self.train else range(801, 901))
        ]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. "
                               "You can use download=True to download it")

    def _check_integrity(self):
        return self.image_folder.exists() and self.image_folder.is_dir()

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(
                url=self._download_url_train
                if self.train else self._download_url_valid,
                extract_root=str(self.dataset_dir),
                download_root=str(self.dataset_dir),
                remove_finished=True)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Image.Image | Tensor, Image.Image | Tensor]:

        image: Image.Image = Image.open(self.image_folder /
                                        self.image_files[idx]).convert("RGB")

        if self.sample_target_generator is not None:
            return self.sample_target_generator(image)

        if self.common_transform is not None:
            image = self.common_transform(image)

        if isinstance(image, Image.Image):
            target = image.copy()  # image is a PIL image
        else:
            target = image.clone()  # image is a tensor

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
