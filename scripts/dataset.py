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
import utils


class PlateDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.images = list(sorted(os.listdir(self.image_dir)))
        self.labels = list(sorted(os.listdir(self.label_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = utils.load_image_from_file(image_path, dtype="float")
        image_width, image_height = image.shape[1], image.shape[0]
        boxes, labels = utils.boxes_and_labels_from_file(label_path, image_height, image_width)

        if self.transform is not None:
            labels = np.array(labels, dtype=np.int64)
            return self.transform(image, {"boxes": boxes, "labels": labels})
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}
            return torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32), target


class SpotDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, "images")
        self.label_dir = os.path.join(data_dir, "labels")
        self.images = list(sorted(os.listdir(self.image_dir)))
        self.labels = list(sorted(os.listdir(self.label_dir)))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = utils.load_image_from_file(image_path, dtype="float")
        label = utils.load_image_from_file(label_path, dtype="float")
        label *= 1000
        image, label = self.pad_to_correct_size(image, label)
        if image.shape[0] != label.shape[0]:
            print(f"Image shape is {image.shape} while label shape is {label.shape}")

        if self.transform is not None:
            return self.transform(image, label)
        else:
            image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            return image, label

    def pad_to_correct_size(self, image, label, value=0):
        """
        Pad the images with the given value so that their shape is dividable by 8 on the X and Y axis. Does
        it by adding the minimum number of values to each side.
        """
        pad_x = 8 - image.shape[1] % 8
        pad_y = 8 - image.shape[0] % 8
        image = np.pad(image, [(pad_y // 2, pad_y // 2 + pad_y % 2),
                               (pad_x // 2, pad_x // 2 + pad_x % 2), (0, 0)], constant_values=value)
        label = np.pad(label, [(pad_y // 2, pad_y // 2 + pad_y % 2),
                               (pad_x // 2, pad_x // 2 + pad_x % 2)], constant_values=value)
        return image, label


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


if __name__ == '__main__':
    from transforms import PlateAlbumentation, CounterAlbumentation
    import matplotlib.pyplot as plt
    dataset_folder = "data/phage_spots/"
    dataset = SpotDataset(dataset_folder, transform=CounterAlbumentation(3))
    image, label = dataset[0]

    plt.figure()
    plt.imshow(image.cpu().numpy().transpose(1, 2, 0))
    plt.figure()
    plt.imshow(label.cpu().numpy()[0])

    plt.show()
