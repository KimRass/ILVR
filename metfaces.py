# Source: https://github.com/NVlabs/metfaces-dataset

from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path


class MetFacesDS(Dataset):
    def __init__(self, data_dir, img_size, hflip):
        self.img_size = img_size

        self.img_paths = list((Path(data_dir)).glob("**/*.png"))

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
        image = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(image=np.array(image))["image"]
