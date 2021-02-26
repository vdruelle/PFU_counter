import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import H5Dataset
from utils import plot_image_dot
from model import UNet
from looper import Looper


CELL_DATA_FOLDER = "data/cells/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cell_dataset = {}
for phase in ["train", "valid"]:
    cell_dataset[phase] = H5Dataset(CELL_DATA_FOLDER + phase + ".h5")

dataloader = {}
for phase in ["train", "valid"]:
    dataloader[phase] = torch.utils.data.DataLoader(cell_dataset[phase], batch_size=6, num_workers=6)

writer = SummaryWriter('runs/test_cell_counter')


# Look at the images / labels
# image, label = cell_dataset["train"][0]
# plot_image_dot(image, label)

network = UNet().to(device)
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train_looper = Looper(network, device, loss, optimizer,
                      dataloader["train"], len(cell_dataset["train"]), writer)
valid_looper = Looper(network, device, loss, optimizer,
                      dataloader["valid"], len(cell_dataset["valid"]), writer, validation=True)

for epoch in range(20):
    print(f"Epoch: {epoch}")
    train_looper.run()
    with torch.no_grad():
        valid_looper.run()
    lr_scheduler.step()
