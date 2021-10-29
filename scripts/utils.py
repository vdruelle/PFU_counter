import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torchvision
import pandas as pd


def plot_plate_detector(image, target):
    """
    Takes in torch tensor (on gpu) and plots the predictions of the network.
    """
    colors = ["C0", "C1", "C2", "C3", "C4"]
    plt.figure(figsize=(14, 10))
    image = image.cpu().numpy().transpose((1, 2, 0))
    target = {k: v.cpu() for k, v in target.items()}
    plt.imshow(image)

    if "scores" in target.keys():
        target = clean_plate_detector_output(target)

    for ii in range(len(target["boxes"])):
        box = target["boxes"][ii]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      facecolor="none",
                                      edgecolor=colors[target["labels"][ii].item()],
                                      alpha=0.5))
        if "scores" in target.keys():
            plt.text(box[0], box[1], round(target["scores"][ii].item(), 2),
                     color=colors[target["labels"][ii].item()])


def clean_plate_detector_output(output, iou_threshold=0.15, score_threshold=0.3):
    """
    Takes the output of the PlateDetector network and cleans it to remove boxes that overlap too much or have
    a low score associated.
    """
    idxs = batch_cleanup_boxes(output["boxes"], output["scores"], output["labels"], iou_threshold)
    output_cleaned = {k: v[idxs] for k, v in output.items()}
    idxs = output_cleaned["scores"] >= score_threshold
    output_cleaned = {k: v[idxs] for k, v in output_cleaned.items()}
    return output_cleaned


def plot_plate_data(image, boxes, labels):
    """
    Takes inputs as numpy arrays and plot them.
    """
    colors = ["C0", "C1", "C2", "C3", "C4"]
    plt.figure(figsize=(14, 10))
    plt.imshow(image)

    for ii in range(len(boxes)):
        box = boxes[ii]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      facecolor="none",
                                      edgecolor=colors[labels[ii]],
                                      alpha=0.5))


def plot_image_dot(image, label):
    im = np.transpose(image, (1, 2, 0))
    lab = np.transpose(label, (1, 2, 0))

    plt.figure(figsize=(14, 10))
    plt.imshow(im)

    plt.figure(figsize=(14, 10))
    plt.imshow(lab)
    plt.show()


def cleanup_boxes(boxes, scores, threshold=0.25):
    """
    Removes boxes that overlap by more than the threshold to a higher scoring box.
    """
    return boxes[torchvision.ops.nms(boxes, scores, threshold)]


def batch_cleanup_boxes(boxes, scores, labels, threshold=0.25):
    """
    Removes boxes that overlap (IoU) by more than the threshold. Does it in a batched manner on all boxes.
    Only removes boxes if they overlap and are of the same label.
    """
    idxs = torchvision.ops.batched_nms(boxes, scores, labels, threshold)
    return idxs


def plot_counter(image, prediction, label):
    """
    Plots a phage colony image and its prediction from the network side by side.
    """
    from matplotlib.colors import LogNorm
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].imshow(image, interpolation=None)
    axs[1].imshow(prediction, cmap="hot", vmin=0, vmax=2, interpolation=None)
    axs[1].set_xlabel(f"Real: {round(np.sum(label)/1000)}   Estimated: {round(np.sum(prediction)/1000)}")
    axs[2].imshow(label, cmap="hot", vmin=0, vmax=2, interpolation=None)


def plot_counter_albu(image, label, raw_image, raw_label):
    """
    Plots a phage colony image and its label side by side. Used to test albumentation augmentation.
    """
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs[0, 0].imshow(raw_image, interpolation=None)
    axs[0, 1].imshow(image, interpolation=None)
    axs[1, 0].imshow(raw_label, cmap="hot", vmin=0, vmax=2, interpolation=None)
    axs[1, 1].imshow(label, cmap="hot", vmin=0, vmax=2, interpolation=None)
    axs[1, 1].set_xlabel(f"Real: {round(np.sum(raw_label)/1000)}   Estimated: {round(np.sum(label)/1000)}")
    plt.tight_layout()
    # plt.show()


def load_image_from_file(path, dtype="float"):
    """
    Loads an image from the given path and rotate it so that plt.imshow shows it in the correct manner.
    It also normalizes the image to be in the range [0,1] for each value.
    Returns the image as a numpy array of shape [height, width, color_channels].
    """
    assert dtype in ["int", "float"], "datatype must be int or float"
    from PIL import Image
    image = Image.open(path)
    image = np.asarray(image, dtype=np.uint8)
    if dtype == "float":
        return image.astype(np.float32) / 255
    else:
        return image


def combine_label_files(original_label_folder, additional_label_folder, combined_label_folder):
    """
    Combines the label files with the same names from orginal and additional folder. Save the combined labels
    in the combined folder.
    """
    import os
    import pandas as pd
    for file in os.listdir(original_label_folder):
        original_file = original_label_folder + file
        additional_file = additional_label_folder + file
        combined_file = combined_label_folder + file

        odf = pd.read_csv(original_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
        adf = pd.read_csv(additional_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
        cdf = pd.concat([odf, adf])

        cdf.to_csv(combined_file, sep=" ", header=False, index=False)


def boxes_and_labels_from_file(label_file, image_height, image_width):
    """
    Creates the boxes and labels list from the label_file.
    """
    df = pd.read_csv(label_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
    df["label"] += 1  # label 0 must be background
    df2 = label_relative_to_absolute(df, image_width, image_height)
    boxes = df2[["x1", "y1", "x2", "y2"]].values.tolist()
    labels = df["label"].values.tolist()

    return boxes, labels


def label_relative_to_absolute(df, image_width, image_height):
    "Converts df containing labels from relative values (cx cy w h) to absolute values (x1 y1 x2 y1)"
    df2 = df.copy(deep=True)
    df2.columns = ["label", "x1", "y1", "x2", "y2"]
    df2["x1"] = (df["cx"] - df["w"] / 2.0) * image_width
    df2["y1"] = (df["cy"] - df["h"] / 2.0) * image_height
    df2["x2"] = (df["cx"] + df["w"] / 2.0) * image_width
    df2["y2"] = (df["cy"] + df["h"] / 2.0) * image_height
    return df2


def label_absolute_to_relative(df, image_width, image_height):
    "Converts df containing labels from absolute values (x1 y1 x2 y2) to relative values (cx cy w h)"
    df2 = df.copy(deep=True)
    df2.columns = ["label", "cx", "cy", "w", "h"]
    df2["cx"] = (df["x1"] + df["x2"]) / (2.0 * image_width)
    df2["cy"] = (df["y1"] + df["y2"]) / (2.0 * image_height)
    df2["w"] = (df["x2"] - df["x1"]) / image_width
    df2["h"] = (df["y2"] - df["y1"]) / image_height
    return df2


def target_from_file(label_file, image_height, image_width):
    """
    Creates the target dictionary need for the RCNN PlateDetector network from the label_file.
    """
    boxes, labels = boxes_and_labels_from_file(label_file, image_height, image_width)
    return {"labels": labels, "boxes": boxes}


def pad_to_correct_size(image, label, value=0):
    """
    Pad the images with the given value so that their shape is dividable by 8 on the X and Y axis. Does
    it by adding the minimum number of values to each side.
    """
    pad_x = 8 - image.shape[1] % 8
    pad_y = 8 - image.shape[0] % 8
    image = np.pad(image, [(pad_y // 2, pad_y // 2 + pad_y % 2),
                           (pad_x // 2, pad_x // 2 + pad_x % 2), (0, 0)], constant_values=value)
    label = np.pad(label, [(pad_y // 2, pad_y // 2 + pad_y % 2),
                           (pad_x // 2, pad_x // 2 + pad_x % 2)], constant_values=value)
    return image, label
