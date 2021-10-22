import os
import sys
import pathlib
import numpy as np
import torch
import torchvision
import utils
import math

from dataset import PlateDataset
from transforms import PlateAlbumentation
from torch.utils.tensorboard import SummaryWriter
from model import PlateDetector


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/PlateDetector_optimized_test')

    dataset_folder = {"train": "data/plates_minimal_spot/", "test": "data/plates_minimal_spot/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = PlateDataset(dataset_folder[phase],
                                            PlateAlbumentation(4) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=2, num_workers=4, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=5, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load("model_saves/Relabeled.pt"))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
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

    torch.save(model.state_dict(), "model_saves/Relabeled.pt")
