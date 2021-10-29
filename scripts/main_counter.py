import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import SpotDataset
from utils import plot_image_dot, plot_counter
from model import UNet
from looper import Looper
from transforms import CounterAlbumentation


def train_phage_data():
    """
    Loads the pretrained network, trains it on the phae colonies data and save the state of the network at the
    end of training.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    writer = SummaryWriter('runs/Counter_new')

    dataset_folder = {"train": "data/phage_spots/train/",
                      "test": "data/phage_spots/test/"}

    phage_colonies_dataset = {}
    for phase in ["train", "test"]:
        phage_colonies_dataset[phase] = SpotDataset(dataset_folder[phase],
                                                    CounterAlbumentation(3) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            phage_colonies_dataset[phase], batch_size=1, num_workers=4, shuffle=(phase == "train"))

    network = UNet().to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(phage_colonies_dataset["train"]), writer)
    test_looper = Looper(network, device, loss, optimizer,
                         dataloader["test"], len(phage_colonies_dataset["test"]), writer, validation=True)

    for epoch in range(70):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            test_looper.run()
        lr_scheduler.step()

    # Saving
    torch.save(network.state_dict(), "model_saves/Counter_phages.pt")


def optimize_counter():
    """
    Loads the pretrained network and does a couple of epoch to test augmentation.
    """
    Phage_colonies_folder = "data/phage_colonies/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))

    phage_colonies_dataset = {}
    for phase in ["train", "valid"]:
        phage_colonies_dataset[phase] = H5Dataset(
            Phage_colonies_folder + phase + "_1000.h5", CounterAlbumentation(phase == "train"))

    dataloader = {}
    for phase in ["train", "valid"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            phage_colonies_dataset[phase], batch_size=6, num_workers=6, shuffle=(phase == "train"))

    writer = SummaryWriter('runs/Optimize_sum')

    network = UNet().to(device)
    network.load_state_dict(torch.load("model_saves/Counter_phages.pt"))

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(phage_colonies_dataset["train"]), writer)
    valid_looper = Looper(network, device, loss, optimizer,
                          dataloader["valid"], len(phage_colonies_dataset["valid"]), writer, validation=True)

    for epoch in range(25):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            valid_looper.run()
        lr_scheduler.step()


def test_network_prediction(network, dataloader, device):
    network.eval()
    im, lab, pred = [], [], []
    for ii, (images, labels) in enumerate(dataloader):
        with torch.no_grad():
            predictions = network(images.to(device))
        predictions = predictions.cpu().numpy()
        predictions = np.squeeze(predictions)
        images = images.cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))
        labels = labels.cpu().numpy()
        labels = np.squeeze(labels)

        if ii == 0:
            im = images
            lab = labels
            pred = predictions
        else:
            im = np.concatenate((images, im))
            lab = np.concatenate((labels, lab))
            pred = np.concatenate((predictions, pred))

    return im, lab, pred


def plot_network_predictions():
    """
    Plots some of the results from the counter.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = UNet().to(device)
    network.load_state_dict(torch.load("model_saves/Counter_phages.pt"))
    Phage_colonies_folder = "data/phage_spots/test/"

    dataset = SpotDataset(Phage_colonies_folder, None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, num_workers=6, shuffle=False)

    images, labels, predictions = test_network_prediction(network, dataloader, device)

    for ii in range(len(images)):
        plot_counter(images[ii], predictions[ii], labels[ii])
    plt.show()


if __name__ == '__main__':
    # pretrain_original_data()
    train_phage_data()
    # optimize_counter()
    # plot_network_predictions()
