from pathlib import Path
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import cv2
import random
from PIL import Image
from torchvision.transforms import functional as TF


### rotate and flip
class Augment_RGB_torch:

    def __init__(self):
        pass

    def transform0(self, torch_tensor):
        return torch_tensor

    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1, -2])
        return torch_tensor

    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1, -2])
        return torch_tensor

    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1, -2])
        return torch_tensor

    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor

    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1, -2])).flip(-2)
        return torch_tensor

    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1, -2])).flip(-2)
        return torch_tensor


def is_image_file(filename):
    return any(
        filename.endswith(extension)
        for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def load_img(filepath):
    # img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    # img = img.astype(np.float32)
    # img = img / 255.
    img = np.asarray(Image.open(filepath).convert("RGB")) / 255.

    return img


augment = Augment_RGB_torch()
transforms_aug = [
    method for method in dir(augment) if callable(getattr(augment, method))
    if not method.startswith('_')
]


class GoProTrainDataset(Dataset):

    def __init__(self, rgb_dir, patch_size: int, target_transform=None):
        super().__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [
            os.path.join(rgb_dir, gt_dir, x)
            for x in clean_files
            if is_png_file(x)
        ]
        self.noisy_filenames = [
            os.path.join(rgb_dir, input_dir, x)
            for x in noisy_files
            if is_png_file(x)
        ]

        self.patch_size: int = patch_size

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = torch.from_numpy(
            np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(
            np.float32(load_img(self.noisy_filenames[tar_index])))

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.patch_size
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H - ps == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return noisy, clean, noisy_filename, clean_filename


class GoProValDataset(Dataset):

    def __init__(self, rgb_dir, patch_size: int, rgb_dir2=None):
        super().__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))

        self.inp_filenames = [
            os.path.join(rgb_dir, 'input', x)
            for x in inp_files
            if is_png_file(x)
        ]
        self.tar_filenames = [
            os.path.join(rgb_dir, 'groundtruth', x)
            for x in tar_files
            if is_png_file(x)
        ]

        # self.img_options = img_options
        self.tar_size = len(self.tar_filenames)  # get the size of target

        # self.ps = self.img_options[
        #     'patch_size'] if img_options is not None else None
        self.ps = patch_size

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))  # type: ignore
            tar_img = TF.center_crop(tar_img, (ps, ps))  # type: ignore

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img  # , filename


if __name__ == "__main__":
    import config as cfg
    ds = GoProTrainDataset(cfg.DATA_PATH / "GoPro/test", 256)
