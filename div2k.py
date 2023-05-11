from torch.utils import data
from skimage import io
from pathlib import Path
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from typing import List, Tuple
from PIL import Image


class DIV2K(VisionDataset):

    _download_url_train = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    _download_url_valid = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

    def __init__(
        self,
        root_dir: Path,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = False,
    ):
        super().__init__(str(root_dir),
                         transform=transform,
                         target_transform=target_transform)
        self.train = train
        self.root_dir: Path = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.folder_basename: str = "DIV2K_train_HR" if self.train else "DIV2K_valid_HR"

        self.image_folder = Path(root_dir) / "DIV2K" / self.folder_basename

        print(self.image_folder)
        print(self.image_folder.exists())

        self.image_files: List[str] = [
            f"{i:04d}.png" for i in range(1, (801 if self.train else 101))
        ]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

    def _check_integrity(self):
        return self.image_folder.exists() and self.image_folder.is_dir()

    def download(self):
        if not self._check_integrity():
            download_and_extract_archive(url=self._download_url_train,
                                         download_root=str(self.root_dir),
                                         extract_root=str(self.image_folder),
                                         remove_finished=True)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        image: Image.Image = Image.open(self.image_folder /
                                        self.image_files[idx]).convert("RGB")
        target = image.copy()
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
