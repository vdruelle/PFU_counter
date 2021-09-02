import os
import sys
import pathlib
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import utils

from dataset import LabDataset, LabH5Dataset


def collate_fn(batch):
    return tuple(zip(*batch))


# use the GPU if is_available
device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
cpu_device = torch.device("cpu")
writer = SummaryWriter('runs/Original')

dataset_folder = "data/phage_plates/"
plate_dataset = {}
for phase in ["train", "valid"]:
    plate_dataset[phase] = LabH5Dataset(dataset_folder + phase + ".h5", None)

dataloader = {}
for phase in ["train", "valid"]:
    dataloader[phase] = torch.utils.data.DataLoader(
        plate_dataset[phase], batch_size=4, num_workers=4, shuffle=False, collate_fn=collate_fn)

image_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good"
label_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good_labels"
dataset = LabDataset(image_dir, label_dir, transform=None)

dataloader2 = {}
dataloader2["train"] = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)


image_original, target_original = iter(dataloader2["train"]).next()
image_new, target_new = iter(dataloader["train"]).next()

image_original = image_original[0]
target_original = target_original[0]
image_new = image_new[0]
target_new = target_new[0]

utils.plot_image_target(image_original, target_original)
utils.plot_image_target(image_new, target_new)
