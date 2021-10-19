import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torchvision


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
        idxs = batch_cleanup_boxes(target["boxes"], target["scores"], target["labels"], 0.15)
    else:
        idxs = range(len(target["boxes"]))

    for ii in idxs:
        box = target["boxes"][ii]
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      facecolor="none",
                                      edgecolor=colors[target["labels"][ii].item()],
                                      alpha=0.5))
        if "scores" in target.keys():
            plt.text(box[0], box[1], round(target["scores"][ii].item(), 2),
                     color=colors[target["labels"][ii].item()])


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


def load_image_from_file(path):
    """
    Loads an image from the given path and rotate it so that plt.imshow shows it in the correct manner.
    It also normalizes the image to be in the range [0,1] for each value.
    Returns the image as a numpy array of shape [height, width, color_channels].
    """
    from PIL import Image
    image = Image.open(path)
    image = image.transpose(Image.ROTATE_270)  # Because PIL consider longest side to be width
    image = np.array(image) / 255
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
