# Source: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ

from torch.utils.data import Dataset
from torchvision.datasets import CelebA
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class CelebADS(Dataset):
    def __init__(self, data_dir, split, img_size, hflip):
        self.ds = CelebA(root=data_dir, split=split, download=True)

        transforms = [
            A.HorizontalFlip(p=0.5),
            A.SmallestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
        if not hflip:
            transforms = transforms[1:]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return self.transform(image=np.array(image))["image"]
