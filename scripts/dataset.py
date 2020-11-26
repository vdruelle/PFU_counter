import numpy as np
import pandas as pd
import os
import pathlib
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class LabDataset(data.Dataset):
    def __init__(self, image_dir, label_dir, use_augmentation=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = list(sorted(os.listdir(image_dir)))
        self.labels = list(sorted(os.listdir(label_dir)))
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(image_path)
        image = image.transpose(Image.ROTATE_270)
        image = image.convert("RGB")
        image_width, image_height = image.size
        # This is in relative coordinate
        df = pd.read_csv(label_path, sep=" ", names=["label", "cx","cy","w","h"])
        df2 = df.copy(deep=True)
        df2.columns = ["label", "x1", "y1", "x2", "y2"]
        df2["x1"] = (df["cx"] - df["w"]/2.0)*image_width
        df2["y1"] = (df["cy"] - df["h"]/2.0)*image_height
        df2["x2"] = (df["cx"] + df["w"]/2.0)*image_width
        df2["y2"] = (df["cy"] + df["h"]/2.0)*image_height
        boxes = df2[["x1", "y1", "x2", "y2"]].values.tolist()
        labels = df["label"].values.tolist()

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        breakpoint()

        target = {}
        target["labels"]
        return image, target
