import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import albumentations as A

from dataset import H5Dataset
from utils import plot_image_dot, plot_counter
from model import UNet
from looper import Looper
from transforms import Compose, CounterRandomHorizontalFlip, CounterRandomVerticalFlip, CounterNormalize


def pretrain_original_data():
    """
    Creates the network, trains it on the original data and save the state of the network at the end of
    training.
    """

    CELL_DATA_FOLDER = "data/cells/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # It uses the transforms
    transform = Compose([
        CounterRandomHorizontalFlip(0.5),
        CounterRandomVerticalFlip(0.5),
    ])

    cell_dataset = {}
    for phase in ["train", "valid"]:
        cell_dataset[phase] = H5Dataset(CELL_DATA_FOLDER + phase + ".h5", transform)

    dataloader = {}
    for phase in ["train", "valid"]:
        dataloader[phase] = torch.utils.data.DataLoader(cell_dataset[phase], batch_size=6, num_workers=6)

    writer = SummaryWriter('runs/cell_counter_default')

    # Look at the images / labels
    # image, label = cell_dataset["train"][0]
    # plot_image_dot(image, label)
    # breakpoint()

    network = UNet().to(device)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(cell_dataset["train"]), writer)
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader["valid"], len(cell_dataset["valid"]), writer, validation=True)

    for epoch in range(50):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            valid_looper.run()
        lr_scheduler.step()

    # Saving
    torch.save(network.state_dict(), "model_saves/Counter_original.pt")


def train_phage_data():
    """
    Loads the pretrained network, trains it on the phae colonies data and save the state of the network at the
    end of training.
    """
    Phage_colonies_folder = "data/phage_colonies/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))

    # It uses the transforms
    transform = Compose([
        CounterRandomHorizontalFlip(0.5),
        CounterRandomVerticalFlip(0.5),
        # CounterNormalize([0.6822, 0.6307, 0.5624], [0.08827, 0.07015, 0.06377])  # from get_normalisation
    ])

    phage_colonies_dataset = {}
    for phase in ["train", "valid"]:
        phage_colonies_dataset[phase] = H5Dataset(Phage_colonies_folder + phase + ".h5", transform)

    # image, label = phage_colonies_dataset["train"][0]
    # plot_image_dot(image, label)
    # breakpoint()

    dataloader = {}
    for phase in ["train", "valid"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            phage_colonies_dataset[phase], batch_size=6, num_workers=6, shuffle=(phase == "train"))

    writer = SummaryWriter('runs/Phage_colonies_png2')

    network = UNet().to(device)
    # network.load_state_dict(torch.load("model_saves/Counter_original.pt"))

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
    torch.save(network.state_dict(), "model_saves/Counter_phages.pt")


def test_network_prediction(network, dataloader, device):
    images, labels = iter(dataloader).next()
    network.eval()
    with torch.no_grad():
        predictions = network(images.to(device))
    predictions = predictions.cpu().numpy()
    predictions = np.squeeze(predictions)
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    labels = labels.cpu().numpy()
    labels = np.squeeze(labels)
    return images, labels, predictions


def plot_network_predictions():
    """
    Plots some of the results from the counter.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = UNet().to(device)
    network.load_state_dict(torch.load("model_saves/Counter_phages.pt"))
    Phage_colonies_folder = "data/phage_colonies/"

    dataset = H5Dataset(Phage_colonies_folder + "train.h5", None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=6, shuffle=True)

    images, labels, predictions = test_network_prediction(network, dataloader, device)

    for ii in range(len(images)):
        plot_counter(images[ii], predictions[ii], labels[ii])
    plt.show()


if __name__ == '__main__':
    # pretrain_original_data()
    train_phage_data()
    # plot_network_predictions()
