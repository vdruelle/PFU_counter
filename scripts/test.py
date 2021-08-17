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
from transforms import Compose, CounterRandomHorizontalFlip, CounterRandomVerticalFlip


Phage_colonies_folder = "data/phage_colonies/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# It uses the transforms
transform = Compose([CounterRandomHorizontalFlip(0.5), CounterRandomVerticalFlip(0.5)])

phage_colonies_dataset = {}
for phase in ["train", "valid"]:
    phage_colonies_dataset[phase] = H5Dataset(Phage_colonies_folder + phase + ".h5", transform)

dataloader = {}
for phase in ["train", "valid"]:
    dataloader[phase] = torch.utils.data.DataLoader(
        phage_colonies_dataset[phase], batch_size=6, num_workers=6)

writer = SummaryWriter('runs/Phage_colonies_pretrained')

network = UNet().to(device)
network.load_state_dict(torch.load("model_saves/Counter_original.pt"))
# network.eval()

loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train_looper = Looper(network, device, loss, optimizer,
                      dataloader["train"], len(phage_colonies_dataset["train"]), writer)
valid_looper = Looper(network, device, loss, optimizer,
                      dataloader["valid"], len(phage_colonies_dataset["valid"]), writer, validation=True)

for epoch in range(50):
    print(f"Epoch: {epoch}")
    train_looper.run()
    with torch.no_grad():
        valid_looper.run()
    lr_scheduler.step()

# Saving
# torch.save(network.state_dict(), "model_saves/Counter_original.pt")
