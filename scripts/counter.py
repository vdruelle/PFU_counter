import utils
import math
from model import PlateDetector
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
from torch.utils.tensorboard import SummaryWriter

from dataset import BoxDataset
from transforms import CounterBoxAlbumentation


def collate_fn(batch):
    return tuple(zip(*batch))


def train_colony_detection():
    """
    Train a FasterRCNN to do colony detection to determine concentration of a dilution spot.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    writer = SummaryWriter('runs/Colony_counter')

    dataset_folder = {"train": "data/phage_spots/subset/train/",
                      "test": "data/phage_spots/subset/test/"}
    plate_dataset = {}
    for phase in ["train", "test"]:
        plate_dataset[phase] = BoxDataset(dataset_folder[phase], CounterBoxAlbumentation(3))

    dataloader = {}
    for phase in ["train", "test"]:
        dataloader[phase] = torch.utils.data.DataLoader(
            plate_dataset[phase], batch_size=6, num_workers=6, shuffle=(phase == "train"),
            collate_fn=collate_fn)

    # The model
    model = PlateDetector(num_classes=2, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
        min_loss = 1e10
        with torch.no_grad():
            valid_loss = []
            for images, targets in dataloader["test"]:
                images = list(image.to(device) for image in images)
                targets = [{key: t[key].to(device) for key in t.keys()} for t in targets]

                losses_dict = model(images, targets)
                valid_loss += [sum(loss for loss in losses_dict.values()).item()]

            valid_loss = np.mean(valid_loss)
            writer.add_scalar("Total_loss/test", valid_loss, epoch)

            if epoch > 10 and valid_loss < min_loss:
                torch.save(model.state_dict(), "model_saves/Colony_counter.pt")
                min_loss = min(valid_loss, min_loss)

        lr_scheduler.step()
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
    model = PlateDetector(num_classes=2, backbone="mobilenet", trainable_backbone_layers=None)
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
        idxs = output["scores"] >= 0.4
        output = {k: v[idxs] for k, v in output.items()}
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
            utils.plot_spot_boxes(image, output)
            plt.show()


def export_to_onnx(model_path, output_path):
    """
    Exports the plate_analyzer network to the ONXX format.
    """
    model = PlateDetector(num_classes=2, backbone="mobilenet", trainable_backbone_layers=None)
    # model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dummy_input = [torch.rand(3, 400, 400)]
    input_names = ["Spot_image"]
    output_names = ["Predicted_dict"]

    torch.onnx.export(model, dummy_input, output_path, verbose=True,
                      input_names=input_names, output_names=output_names, opset_version=11)


def test_onnx(model_path, input_path):
    "Tests if the onnx format save works."
    import onnxruntime as ort
    from PIL import Image
    im = Image.open(input_path)
    im = im.resize((400, 400))
    x = np.asarray(im, dtype=np.uint8)
    x = x.astype(np.float32) / 255
    x = np.transpose(x, (2, 0, 1))
    ort_sess = ort.InferenceSession(model_path)
    outputs = ort_sess.run(None, {'Spot_image': x})
    print(outputs)


if __name__ == '__main__':
    # train_colony_detection()
    predict_full_dataset("model_saves/Colony_counter_newdata.pt", "data/phage_spots/subset/test/images/",
                         "data/phage_spots/subset/test/labels/", show=True)
    # export_to_onnx("model_saves/Colony_counter.pt", "model_saves/Colony_counter.onnx")
    # test_onnx("model_saves/Colony_counter.onnx", "data/phage_spots/subset/test/images/20211007_105802_6.jpg")
