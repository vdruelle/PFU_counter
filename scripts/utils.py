import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np


def plot_image_target(image, target):
    colors = ["C0", "C1", "C2", "C3", "C4"]
    plt.figure(figsize=(14, 10))
    image = image.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image)

    # boxes = []
    for ii, box in enumerate(target["boxes"]):
        plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      facecolor="none",
                                      edgecolor=colors[target["labels"][ii].item()],
                                      alpha=0.5))
        # boxes += [Rectangle((box[0], box[1]), box[2] - box[0], box[3] -
        #                     box[1], facecolor=None, edgecolor=colors[target["labels"][ii].item()])]
        if "scores" in target.keys():
            plt.text(box[0], box[1], round(target["scores"][ii].item(), 2),
                     color=colors[target["labels"][ii].item()])
    # pc = PatchCollection(boxes)
    # plt.gca().add_collection(pc)
    # plt.show()


def plot_image_dot(image, label):
    im = np.transpose(image, (1, 2, 0))
    lab = np.transpose(label, (1, 2, 0))

    plt.figure(figsize=(14, 10))
    plt.imshow(im)

    plt.figure(figsize=(14, 10))
    plt.imshow(lab)
    plt.show()


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
