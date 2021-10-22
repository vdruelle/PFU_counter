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


def train_small_dataset(dataset_folder, writer_name, model_save_name, num_classes=4):
    """
    Trains a model based on a small number of labeled images and saves the model.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/' + writer_name)
    dataset_folder = {"train": dataset_folder, "test": dataset_folder}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = PlateDataset(dataset_folder[phase],
                                            PlateAlbumentation(5) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=2, num_workers=4, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=num_classes + 1, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    num_epochs = 45
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

    torch.save(model.state_dict(), "model_saves/" + model_save_name + ".pt")


def predict_full_dataset(model_save_path, image_folder, output_label_folder, num_classes=4):
    """
    Uses a minimally trained model to predict the labels of the full dataset.
    image_folder is the folder containing all the images from which to predict the labels.
    """
    import pandas as pd

    os.makedirs(output_label_folder, exist_ok=True)
    image_list = list(sorted(os.listdir(image_folder)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    model = PlateDetector(num_classes=num_classes + 1, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    for image_name in image_list:
        # Image loading
        image_path = os.path.join(image_folder, image_name)
        image = utils.load_image_from_file(image_path)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        image.to(device)

        # Predictions
        image = image.to(device)
        with torch.no_grad():
            outputs = model([image])
        output = outputs[0]

        image_width, image_height = image.shape[2], image.shape[1]
        df = pd.DataFrame(columns=["label", "x1", "y1", "x2", "y2"])
        df["label"] = output["labels"].tolist()
        df[["x1", "y1", "x2", "y2"]] = output["boxes"].tolist()

        df2 = utils.label_absolute_to_relative(df, image_width, image_height)

        return image, output


if __name__ == '__main__':
    # data_folder = "data/plates_labeled_minimal/spot_labeling/"
    # writer_name = "PlateDetector_minimal_spot"
    # model_save_name = "Plate_detector_spots"
    # train_small_dataset(data_folder, writer_name, model_save_name, num_classes=3)

    model_save = "model_saves/Plate_detector_spots.pt"
    image_folder = "data/plates_labeled/spot_labeling/images"
    output_folder = "data/plates_labeled/spot_labeling/labels"
    predict_full_dataset(model_save, image_folder, output_folder, num_classes=3)
