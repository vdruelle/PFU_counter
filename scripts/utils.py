import matplotlib.pyplot as plt
import numpy as np


def plot_image_target(image, target):
    plt.figure(figsize=(14, 10))
    image = image.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image)

    boxes = []
    for box in target["boxes"]:
        boxes += [Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1])]
    pc = PatchCollection(boxes, facecolor="none", edgecolor="blue", alpha=0.5)
    plt.gca().add_collection(pc)
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
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    axs[0].imshow(image, interpolation=None)
    axs[1].imshow(prediction, cmap="hot", vmin=0, vmax=2, interpolation=None)
    axs[1].set_xlabel(f"Real: {round(np.sum(label)/100)}   Estimated: {round(np.sum(prediction)/100)}")
    axs[2].imshow(label, cmap="hot", vmin=0, vmax=2, interpolation=None)
