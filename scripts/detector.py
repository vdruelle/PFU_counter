import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import BoxDataset
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
    writer = SummaryWriter('runs/test')

    dataset_folder = {"train": "data/plates_labeled/train/",
                      "test": "data/plates_labeled/test/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = BoxDataset(dataset_folder[phase],
                                          PlateAlbumentation(4) if phase == "train" else None)

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=4, num_workers=4, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=4, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load("model_saves/Plate_detector.pt"))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    num_epochs = 30
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
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for images, targets in dataloader["test"]:
                images = list(image.to(device) for image in images)
                targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

                predictions = model(images)
                valid_loss += compute_validation_errors(predictions, targets)

            valid_loss /= len(plate_dataset["test"])
            writer.add_scalar("Total_loss/test", valid_loss, epoch)

        lr_scheduler.step()

    # torch.save(model.state_dict(), "model_saves/Plate_detector_new.pt")
    print("That's it!")


def predict_full_dataset(model_save_path, image_folder, output_label_folder, show=False):
    """
    Uses a trained model to predict the labels of the full dataset.
    image_folder is the folder containing all the images from which to predict the labels.
    When show is True, just plots the results instead of saving them.
    """
    import pandas as pd

    os.makedirs(output_label_folder, exist_ok=True)
    image_list = list(sorted(os.listdir(image_folder)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))

    model = PlateDetector(num_classes=4, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    for ii, image_name in enumerate(image_list):
        # Image loading
        print(f"Predicting image {ii} of {len(image_list)}")
        image_path = os.path.join(image_folder, image_name)
        image = utils.load_image_from_file(image_path)
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)

        # Predictions
        image = image.to(device)
        with torch.no_grad():
            outputs = model([image])
        output = outputs[0]
        output = utils.clean_plate_detector_output(output, 0.15, 0.3)
        boxes, labels = output["boxes"], output["labels"]

        if not show:
            image_width, image_height = image.shape[2], image.shape[1]
            df = pd.DataFrame(columns=["label", "x1", "y1", "x2", "y2"])
            df["label"] = labels.tolist()
            df[["x1", "y1", "x2", "y2"]] = boxes.tolist()

            df2 = utils.label_absolute_to_relative(df, image_width, image_height)
            label_name = os.path.join(output_label_folder, image_name[:-4] + ".txt")
            df2 = df2.sort_values(by=["label"])
            df2["label"] = df2["label"] - 1  # Because 0 is background for the RCNN
            df2.to_csv(label_name, sep=" ", header=False, index=False)
        else:
            utils.plot_plate_detector(image, output)
            plt.show()


def compute_validation_errors(predictions, targets):
    """
    Validation error computed using Intersection over Union loss.
    """
    error = 0
    for prediction, target in zip(predictions, targets):
        prediction = utils.clean_plate_detector_output(prediction, 0.15, 0.3)
        # Checks for the IoU error with the closest match for each predicted box
        error += torch.sum(1 - torchvision.ops.generalized_box_iou(
            prediction["boxes"], target["boxes"]).max(dim=0)[0])

        # Add error if a box is missing
        if not torch.any(prediction["labels"] == 1):  # Missing plate_name
            error += 2
        if not torch.any(prediction["labels"] == 2):  # Missing phage_names
            error += 2

        nb_box_pred = len(torch.where(prediction["labels"] == 3)[0])
        nb_box_target = len(torch.where(target["labels"] == 3)[0])
        if nb_box_pred < nb_box_target:
            error += (nb_box_target - nb_box_pred)  # less penality for missing spot than plate/phage name

    return error


def export_to_onnx(model_path, output_path):
    """
    Exports the plate_analyzer network to the ONXX format.
    """
    model = PlateDetector(num_classes=4, backbone="mobilenet", trainable_backbone_layers=None)
    # model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = [torch.rand(3, 4608, 3456)]
    input_names = ["Plate_image"]
    output_names = ["Predicted_dict"]

    torch.onnx.export(model, dummy_input, output_path, verbose=True,
                      input_names=input_names, output_names=output_names, opset_version=11)


def test_onnx(model_path):
    "Tests if the onnx format save works."
    import onnxruntime as ort
    x = torch.rand(3, 4608, 3456).numpy()
    ort_sess = ort.InferenceSession('model_saves/Plate_detector.onnx')
    outputs = ort_sess.run(None, {'Plate_image': x})
    print(outputs)


if __name__ == '__main__':
    # train_plate_detection()
    # predict_full_dataset("model_saves/Plate_detector_new.pt", "data/plates_labeled/to_label/images/",
    #                      "data/plates_labeled/to_label/labels/", show=False)
    # export_to_onnx("model_saves/Plate_detector.pt", "model_saves/Plate_detector.onxx")
    # test_onnx("model_saves/Plate_detector.onnx")
    train_test()
