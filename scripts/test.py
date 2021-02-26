import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import H5Dataset
from utils import plot_image_dot
from model import UNet


CELL_DATA_FOLDER = "data/cells/"

cell_dataset = {}
for phase in ["train", "valid"]:
    cell_dataset[phase] = H5Dataset(CELL_DATA_FOLDER + phase + ".h5")

# Look at the images / labels
# image, label = cell_dataset["train"][0]
# plot_image_dot(image, label)

network = Unet()
