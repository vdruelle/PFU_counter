import h5py
from random import random
import numpy as np
import pandas as pd
import os
import pathlib
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
from typing import Optional


class LabDataset(data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = list(sorted(os.listdir(image_dir)))
        self.labels = list(sorted(os.listdir(label_dir)))
        self.transform = transform

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
        df = pd.read_csv(label_path, sep=" ", names=["label", "cx", "cy", "w", "h"])
        df["label"] += 1  # label 0 must be background
        df2 = df.copy(deep=True)
        df2.columns = ["label", "x1", "y1", "x2", "y2"]
        df2["x1"] = (df["cx"] - df["w"] / 2.0) * image_width
        df2["y1"] = (df["cy"] - df["h"] / 2.0) * image_height
        df2["x2"] = (df["cx"] + df["w"] / 2.0) * image_width
        df2["y2"] = (df["cy"] + df["h"] / 2.0) * image_height
        boxes = df2[["x1", "y1", "x2", "y2"]].values.tolist()
        labels = df["label"].values.tolist()
        area = (df["w"] * image_width * df["h"] * image_height).values.tolist()

        image = transforms.functional.to_tensor(image)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(area)

        target = {}
        target["labels"] = labels
        target["boxes"] = boxes
        target["image_id"] = image_id
        target["area"] = area

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target


class LabH5Dataset(data.Dataset):
    def __init__(self, dataset_path, transform=None):
        super(LabH5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']
        self.boxes = self.h5['boxes']
        self.labels = self.h5['labels']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Format the image, boxes and labels and return a torch tensor for the image, and a dictionary of torch
        tensors for the target, as expected by the RCNN network.
        """
        # Removing the padding with -1
        boxes, labels = self.remove_padding(self.boxes[index], self.labels[index])

        if self.transform is not None:
            labels = labels.astype(np.int64)
            return self.transform(self.images[index], {"boxes": boxes, "labels": labels})
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
            return torch.tensor(np.transpose(self.images[index], (2, 0, 1))), target

    def remove_padding(self, boxes, labels):
        """
        Removes the -1 padding added so that they have the same dimension and fit into an h5 dataset.
        """
        box = np.delete(boxes, np.where(boxes == -1)[0], axis=0)
        lab = labels[labels != -1]
        return box, lab

# This part if from the github on cell counting.


class H5Dataset(data.Dataset):
    """PyTorch dataset for HDF5 files generated with `get_data.py`."""

    def __init__(self,
                 dataset_path: str,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize flips probabilities and pointers to a HDF5 file.
        Args:
            dataset_path: a path to a HDF5 file
        """
        super(H5Dataset, self).__init__()
        self.h5 = h5py.File(dataset_path, 'r')
        self.images = self.h5['images']
        self.labels = self.h5['labels']
        self.transform = transform

    def __len__(self):
        """Return no. of samples in HDF5 file."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return next sample (randomly flipped)."""

        if self.transform is not None:
            return self.transform(self.images[index], self.labels[index])
        else:
            return self.images[index], self.labels[index]

    def get_normalisation(self):
        """
        Returns the 3D mean and std from the list of images.
        """
        data = self.images
        mean = [data[:, ii, :, :].mean() for ii in range(data.shape[1])]
        std = [data[:, ii, :, :].std() for ii in range(data.shape[1])]
        return mean, std


# --- PYTESTS --- #


def test_loader():
    """Test HDF5 dataloader with flips on and off."""
    run_batch(flip=False)
    run_batch(flip=True)


def run_batch(flip):
    """Sanity check for HDF5 dataloader checks for shapes and empty arrays."""
    # datasets to test loader on
    datasets = {
        'data/cells': (3, 256, 256),
    }

    # for each dataset check both training and validation HDF5
    # for each one check if shapes are right and arrays are not empty
    for dataset, size in datasets.items():
        for h5 in ('train.h5', 'valid.h5'):
            # create a loader in "all flips" or "no flips" mode
            data = H5Dataset(os.path.join(dataset, h5),
                             horizontal_flip=1.0 * flip,
                             vertical_flip=1.0 * flip)
            # create dataloader with few workers
            data_loader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=4)

            # take one batch, check samples, and go to the next file
            for img, label in data_loader:
                # image batch shape (#workers, #channels, resolution)
                assert img.shape == (4, *size)
                # label batch shape (#workers, 1, resolution)
                assert label.shape == (4, 1, *size[1:])

                assert torch.sum(img) > 0
                assert torch.sum(label) > 0

                break
