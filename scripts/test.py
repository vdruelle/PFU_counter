import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from dataset import LabDataset
from transforms import RandomHorizontalFlip, Compose, GaussianBlur


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_image_target(image, target):
    plt.figure(figsize=(14, 10))
    image = image.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image)

    boxes = []
    for box in target["boxes"]:
        boxes += [Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1])]
    pc = PatchCollection(boxes, facecolor="none", edgecolor="blue", alpha=0.5)
    plt.gca().add_collection(pc)
    # plt.show()


# use the GPU if is_available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good"
label_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good_labels"

# Could add data augmentation here
transform = Compose([RandomHorizontalFlip(0.5), GaussianBlur(3)])
dataset = LabDataset(image_dir, label_dir, transform=transform)
dataset_test = LabDataset(image_dir, label_dir, transform=None)

# split the dataset in train and test set
test_set = torch.utils.data.Subset(dataset_test, range(9, 14))

image, target = test_set[0]
plot_image_target(image, target)
for ii in range(3):
    img = torchvision.transforms.GaussianBlur(5, (0.1, 2)).forward(image)
    plot_image_target(img, target)
plt.show()
