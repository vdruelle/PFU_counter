import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from dataset import SpotDataset
from utils import plot_image_dot, plot_counter
from model import UNet
from looper import Looper
from transforms import CounterAlbumentation


def train_phage_data(data_folder, scaling=100):
    """
    Loads the pretrained network, trains it on the phae colonies data and save the state of the network at the
    end of training.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    writer = SummaryWriter('runs/Counter_sigmo100')

    dataset_folder = {"train": data_folder + "train/",
                      "test": data_folder + "test/"}

    dataset = {}
    for phase in ["train", "test"]:
        dataset[phase] = SpotDataset(dataset_folder[phase] + "images/",
                                     dataset_folder[phase] + "density_kdtree/",
                                     scaling=scaling,
                                     transform=CounterAlbumentation(2) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"))

    network = UNet().to(device)
    # network.load_state_dict(torch.load("model_saves/Counter_original.pt"))

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(dataset["train"]), writer, scaling=scaling)
    test_looper = Looper(network, device, loss, optimizer,
                         dataloader["test"], len(dataset["test"]), writer, scaling=scaling, validation=True)

    for epoch in range(50):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            test_looper.run()
        lr_scheduler.step()

    # Saving
    torch.save(network.state_dict(), "model_saves/Counter_test.pt")


def optimize_counter():
    """
    Loads the pretrained network and does a couple of epoch to test augmentation.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    writer = SummaryWriter('runs/Counter_sum')

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
    network.load_state_dict(torch.load("model_saves/Counter_phages.pt"))

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-5, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(phage_colonies_dataset["train"]), writer)
    test_looper = Looper(network, device, loss, optimizer,
                         dataloader["test"], len(phage_colonies_dataset["test"]), writer, validation=True)

    for epoch in range(30):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            test_looper.run()
        lr_scheduler.step()

    # Saving
    # torch.save(network.state_dict(), "model_saves/Counter_phages_2.pt")


def plot_network_predictions(model_save, image_folder, density_folder=""):
    """
    Plots some of the results from the counter.
    """
    import utils
    dataset_shape = (296, 304)
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    network = UNet().to(device)
    network.load_state_dict(torch.load(model_save))
    network.eval()

    image_list = list(sorted(os.listdir(image_folder)))

    for ii, image_name in enumerate(image_list):
        image = utils.load_image_from_file(image_folder + image_name)
        image = utils.pad_single_to_shape(image, dataset_shape)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        image = image.to(device)

        with torch.no_grad():
            output = network(torch.unsqueeze(image, 0))
            output[output < 0] = 0

        image = np.transpose(image.cpu().numpy(), (1, 2, 0))
        output = output.cpu().numpy()
        if density_folder == "":
            utils.plot_counter(image, output[0, 0], vmax=10)
        else:
            density = np.load(density_folder + image_name[:-4] + "_labels.npy")
            density = utils.pad_single_to_shape(density, dataset_shape)
            utils.plot_counter_density(image, output[0, 0], density, vmax=1, scaling=100)
            plt.show()


def train_minimal(image_folder, density_folder):
    """
    Trains a counter network on the minimal dataset provided.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    dataset = SpotDataset(image_folder, density_folder, scaling=100, transform=CounterAlbumentation(3))
    writer = SummaryWriter('runs/Small_counter_kdtree_albu3')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    network = UNet().to(device)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    looper = Looper(network, device, loss, optimizer,
                    dataloader, len(dataset), writer, scaling=100, validation=False)

    for epoch in range(80):
        print(f"Epoch: {epoch}")
        looper.run()
        lr_scheduler.step()

    # Saving
    torch.save(network.state_dict(), "model_saves/Small_counter_kdtree_ablu3.pt")


if __name__ == '__main__':
    # train_phage_data("data/phage_spots_old/", scaling=100)
    # optimize_counter()
    plot_network_predictions("model_saves/Counter_test.pt",
                             "data/phage_spots_old/test/images/", "data/phage_spots_old/test/density_kdtree/")
    # train_minimal("data/phage_spots_minimal/dot_labeling/images",
    # "data/phage_spots_minimal/dot_labeling/density_kdtree")
