import os
import sys
import pathlib
import numpy as np
import torch
import torchvision
import utils

from dataset import LabDataset, LabH5Dataset
import pandas as pd


if __name__ == '__main__':
    original_labels = "data/lab_raw_good_labels/"
    additional_labels = "data/lab_raw_good_labels2/"
    combined_labels = "data/lab_raw_good_combined/"

    for file in os.listdir(original_labels):
        original_file = original_labels + file
        additional_file = additional_labels + file
        combined_file = combined_labels + file

        odf = pd.read_csv(original_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
        adf = pd.read_csv(additional_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
        cdf = pd.concat([odf, adf])

        cdf.to_csv(combined_file, sep=" ", header=False, index=False)
