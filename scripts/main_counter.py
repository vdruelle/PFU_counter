import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import SpotDataset
from utils import plot_image_dot, plot_counter
from model import UNet
from looper import Looper
from transforms import CounterAlbumentation


def train_phage_data(data_folder, scaling=1000):
    """
    Loads the pretrained network, trains it on the phae colonies data and save the state of the network at the
    end of training.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    writer = SummaryWriter('runs/Counter_old')

    dataset_folder = {"train": data_folder + "train/",
                      "test": data_folder + "test/"}

    dataset = {}
    for phase in ["train", "test"]:
        dataset[phase] = SpotDataset(dataset_folder[phase] + "images/",
                                     dataset_folder[phase] + "density_kdtree/",
                                     scaling=scaling,
                                     transform=None)
                                     # transform=CounterAlbumentation(3) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"))

    network = UNet().to(device)
    network.load_state_dict(torch.load("model_saves/Counter_original.pt"))

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
    torch.save(network.state_dict(), "model_saves/Counter_old.pt")


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


def plot_network_predictions(model_save, image_folder):
    """
    Plots some of the results from the counter.
    """
    import utils
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    network = UNet().to(device)
    network.load_state_dict(torch.load(model_save))
    network.eval()

    image_list = list(sorted(os.listdir(image_folder)))
    for ii, image_name in enumerate(image_list):
        image = utils.load_image_from_file(image_folder + image_name)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        image = image.to(device)

        output = model([image])

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
    train_phage_data("data/phage_spots_old/")
    # optimize_counter()
    # plot_network_predictions()
    # train_minimal("data/phage_spots_minimal/dot_labeling/images",
    # "data/phage_spots_minimal/dot_labeling/density_kdtree")
