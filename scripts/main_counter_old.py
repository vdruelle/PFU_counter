import utils
import math
from model import PlateDetector
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from dataset import SpotDataset, PlateDataset
from utils import plot_image_dot, plot_counter
from model import UNet
from looper import Looper
from transforms import CounterAlbumentation, PlateAlbumentation


def train_phage_data(data_folder, scaling=100):
    """
    Loads the pretrained network, trains it on the phae colonies data and save the state of the network at the
    end of training.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("Can't use GPU"))
    writer = SummaryWriter('runs/Counter_box2')

    dataset_folder = {"train": data_folder + "train/",
                      "test": data_folder + "test/"}

    dataset = {}
    for phase in ["train", "test"]:
        dataset[phase] = SpotDataset(dataset_folder[phase] + "images/",
                                     dataset_folder[phase] + "density/",
                                     scaling=scaling,
                                     transform=CounterAlbumentation(2) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"))

    network = UNet().to(device)
    network.load_state_dict(torch.load("model_saves/Counter_tanh_100.pt"))

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    train_looper = Looper(network, device, loss, optimizer,
                          dataloader["train"], len(dataset["train"]), writer, scaling=scaling)
    test_looper = Looper(network, device, loss, optimizer,
                         dataloader["test"], len(dataset["test"]), writer, scaling=scaling, validation=True)

    for epoch in range(80):
        print(f"Epoch: {epoch}")
        train_looper.run()
        with torch.no_grad():
            test_looper.run()
        lr_scheduler.step()

    # Saving
    torch.save(network.state_dict(), "model_saves/Counter_box.pt")


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
                                                    CounterAlbumentation(2) if phase == "train" else None)

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
            utils.plot_counter(image, output[0, 0], vmax=1, scaling=50)
            plt.show()
        else:
            density = np.load(density_folder + image_name[:-4] + "_labels.npy")
            # density = np.load(density_folder + image_name[:-4] + ".npy")
            density = utils.pad_single_to_shape(density, dataset_shape)
            utils.plot_counter_density(image, output[0, 0], density, vmax=1, scaling=50)
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


def collate_fn(batch):
    return tuple(zip(*batch))


def train_plate_detection():
    """
    Train a FasterRCNN to do plate element detection using the LabH5Dataset.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/Plate_detector')

    dataset_folder = {"train": "data/phage_spots_minimal/box_labeling/train/",
                      "test": "data/phage_spots_minimal/box_labeling/test/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = PlateDataset(dataset_folder[phase],
                                            PlateAlbumentation(4) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=2, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    num_epochs = 60
    n_iter = 0
    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch} ---")
        # Train
        model.train()
        for ii, (images, targets) in enumerate(dataloader["train"]):
            images = list(image.to(device) for image in images)
            targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

            losses_dict = model(images, targets)
            loss_sum = sum(loss for loss in losses_dict.values())

            # Writting to tensorboard
            for key in losses_dict.keys():
                writer.add_scalar("Losses/" + key, losses_dict[key].item(), n_iter)
            writer.add_scalar("Total_loss/train", loss_sum.item(), n_iter)

            if not math.isfinite(loss_sum):
                print("Loss is {}, stopping training".format(loss_sum.item()))
                sys.exit(1)
            else:
                print(f"Batch: {ii}    Loss: {loss_sum.item()}")

            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            n_iter += 1

        # Test
        with torch.no_grad():
            valid_loss = 0
            for images, targets in dataloader["test"]:
                images = list(image.to(device) for image in images)
                targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

                losses_dict = model(images, targets)
                valid_loss += sum(loss for loss in losses_dict.values())

            valid_loss /= len(plate_dataset["test"])
            writer.add_scalar("Total_loss/test", valid_loss, epoch)

        lr_scheduler.step()

    torch.save(model.state_dict(), "model_saves/Dot_counting.pt")
    print("That's it!")


def predict_plate(model_path, image_path):
    """
    Function to test the predictions of the network trained by train_plate_detection().
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))

    model = PlateDetector(num_classes=2, backbone="mobilenet")
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image = utils.load_image_from_file(image_path)
    image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
    image = image.to(device)

    with torch.no_grad():
        outputs = model([image])
        output = outputs[0]
        utils.plot_plate_detector(image, output)
    plt.show()


if __name__ == '__main__':
    # train_phage_data("data/phage_spots_minimal/box_labeling/", scaling=50)
    # optimize_counter()
    plot_network_predictions("model_saves/Counter_box.pt",
                             "data/phage_spots_minimal/dot_labeling/train/images/")
    # train_minimal("data/phage_spots_minimal/dot_labeling/images",
    # "data/phage_spots_minimal/dot_labeling/density_kdtree")
    # train_plate_detection()
    # predict_plate("model_saves/Dot_counting.pt", "data/phage_spots/all_images/20211007_110618_2.jpg")
