import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets

from dataset import LabDataset

# use the GPU if is_available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good"
label_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good_labels"

dataset = LabDataset(image_dir, label_dir)
dataset.__getitem__(0)
