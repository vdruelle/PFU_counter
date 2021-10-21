import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import PlateDataset, LabH5Dataset
from transforms import PlateAlbumentation
from model import PlateDetector
import utils


def collate_fn(batch):
    return tuple(zip(*batch))


def train_plate_detection():
    """
    Train a FasterRCNN to do plate element detection using the LabH5Dataset.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/Plate_detector_0')

    dataset_folder = {"train": "data/plates_labeled/train/", "test": "data/plates_labeled/test/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = PlateDataset(dataset_folder[phase],
                                            PlateAlbumentation(0) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=5, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    num_epochs = 20
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

    torch.save(model.state_dict(), "model_saves/Plate_detection.pt")
    print("That's it!")


def optimize_plate_detection():
    """
    Train a FasterRCNN to do plate element detection using the LabH5Dataset.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/PlateDetector_optimized_4')

    dataset_folder = {"train": "data/plates_labeled/train/", "test": "data/plates_labeled/test/"}
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
    model = PlateDetector(num_classes=5, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load("model_saves/Plate_detection.pt"))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    num_epochs = 25
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

    torch.save(model.state_dict(), "model_saves/Optimized_4.pt")
    print("That's it!")


def predict_plate(model_path="model_saves/Plate_detection.pt"):
    """
    Function to test the predictions of the network trained by train_plate_detection().
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))

    dataset_folder = {"train": "data/plates_labeled/train/", "test": "data/plates_labeled/test/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = PlateDataset(dataset_folder[phase], None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=1, num_workers=1, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    model = PlateDetector()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader["test"]:
            images = list(image.to(device) for image in images)

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            utils.plot_plate_detector(images[0], outputs[0])
    plt.show()


def compute_validation_errors(predictions, targets):
    """
    Validation error computed using Intersection over Union loss.
    """
    error = 0
    for prediction, target in zip(predictions, targets):
        # For plate name
        idxs_plate_name = torch.where(prediction["labels"] == 1)[0]
        if idxs_plate_name.shape[0] != 0:
            idx = torch.argmax(prediction["scores"][idxs_plate_name])
            tbox = target["boxes"][target["labels"] == 1][0]
            pbox = prediction["boxes"][idxs_plate_name[idx]]
            error += 1 - torchvision.ops.generalized_box_iou(pbox.unsqueeze(0), tbox.unsqueeze(0))

        # For phage names
        idxs_phage_names = torch.where(prediction["labels"] == 2)[0]
        if idxs_phage_names.shape[0] != 0:
            idx = torch.argmax(prediction["scores"][idxs_phage_names])
            tbox = target["boxes"][target["labels"] == 2][0]
            pbox = prediction["boxes"][idxs_phage_names[idx]]
            error += 1 - torchvision.ops.generalized_box_iou(pbox.unsqueeze(0), tbox.unsqueeze(0))

        # For phage columns
        idxs_phage_columns = torch.where(prediction["labels"] == 3)[0]
        if idxs_phage_columns.shape[0] != 0:
            tboxes = target["boxes"][target["labels"] == 3]
            pboxes = prediction["boxes"][idxs_phage_columns]
            # This step removes boxes of lower score that overlap by more than 25% with a higher score box
            pboxes = utils.cleanup_boxes(pboxes, prediction["scores"][idxs_phage_columns], 0.25)
            # Takes the best box for each target box, and compute the IoU error between each pair
            error += torch.sum(1 - torchvision.ops.generalized_box_iou(pboxes, tboxes).max(dim=0)[0])

        # For phage rows
        idxs_phage_rows = torch.where(prediction["labels"] == 4)[0]
        if idxs_phage_rows.shape[0] != 0:
            tboxes = target["boxes"][target["labels"] == 4]
            pboxes = prediction["boxes"][idxs_phage_rows]
            # This step removes boxes of lower score that overlap by more than 25% with a higher score box
            pboxes = utils.cleanup_boxes(pboxes, prediction["scores"][idxs_phage_rows], 0.25)
            # Takes the best box for each target box, and compute the IoU error between each pair
            error += torch.sum(1 - torchvision.ops.generalized_box_iou(pboxes, tboxes).max(dim=0)[0])

    return error


if __name__ == '__main__':
    # train_plate_detection()
    # optimize_plate_detection()
    predict_plate("model_saves/Optimized_0.pt")
