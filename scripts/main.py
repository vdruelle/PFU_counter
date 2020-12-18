import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

from dataset import LabDataset
from transforms import RandomHorizontalFlip


def collate_fn(batch):
    return tuple(zip(*batch))


# use the GPU if is_available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good"
label_dir = str(pathlib.Path.cwd()) + "/data/lab_raw_good_labels"

writer = SummaryWriter('runs/Run_with_augmentation')

# Could add data augmentation here
transform = RandomHorizontalFlip(0.5)
dataset = LabDataset(image_dir, label_dir, transform=transform)
# dataset = LabDataset(image_dir, label_dir, transform=None)

# split the dataset in train and test set
train_set, test_set = torch.utils.data.random_split(dataset, [31, 5])

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

# The model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 4  # background + the 3 others
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replacing the pre rtained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


num_epochs = 10
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    epoch_mean_loss = 0
    model.train()
    for ii, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

        losses_dict = model(images, targets)
        losses_sum = sum(loss for loss in losses_dict.values())
        loss = losses_sum.item()

        n_iter = epoch*len(train_set) + ii
        for key in losses_dict.keys():
            writer.add_scalar("Losses/" + key, losses_dict[key], n_iter)
        writer.add_scalar("Total loss", loss, n_iter)

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            print(losses_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses_sum.backward()
        optimizer.step()

        print(f"Epoch {epoch}: total loss {loss}")

    # update the learning rate
    lr_scheduler.step()

print("That's it!")
