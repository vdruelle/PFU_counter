import os
import sys
import pathlib
import numpy as np
import math
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from dataset import LabDataset, LabH5Dataset
from transforms import RandomHorizontalFlip, Compose, GaussianBlur


def collate_fn(batch):
    return tuple(zip(*batch))


# use the GPU if is_available
device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
writer = SummaryWriter('runs/Default')

dataset_folder = "data/phage_plates/"
plate_dataset = {}
for phase in ["train", "valid"]:
    plate_dataset[phase] = LabH5Dataset(dataset_folder + phase + ".h5", None)

dataloader = {}
for phase in ["train", "valid"]:
    dataloader[phase] = torch.utils.data.DataLoader(
        plate_dataset[phase], batch_size=4, num_workers=4, shuffle=False, collate_fn=collate_fn)

# The model
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
num_classes = 4  # background + the 3 others
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replacing the pre rtained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


num_epochs = 15
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
            writer.add_scalar("Losses/" + key, losses_dict[key].item(), ii)
        writer.add_scalar("Total_loss/train", loss_sum.item(), ii)

        if not math.isfinite(loss_sum):
            print("Loss is {}, stopping training".format(loss_sum.item()))
            sys.exit(1)
        else:
            print(f"Image: {ii}    Loss: {loss_sum.item()}")

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()

    # Test
    with torch.no_grad():
        test_loss = 0
        for images, targets in dataloader["valid"]:
            images = list(image.to(device) for image in images)
            targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

            losses_dict = model(images, targets)
            loss_sum = sum(loss for loss in losses_dict.values())
            test_loss += loss_sum.item()

        test_loss = test_loss / len(plate_dataset["valid"])
        writer.add_scalar("Total_loss/test", test_loss, epoch)

    # update the learning rate
    lr_scheduler.step()

print("That's it!")
