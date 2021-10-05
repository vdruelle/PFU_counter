import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torchvision


def plot_image_target(image, target):
    colors = ["C0", "C1", "C2", "C3", "C4"]
    plt.figure(figsize=(14, 10))
    image = image.cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image)

    # Filtering lox quality overlapping boxes
    # idxs_columns = torch.where(target["labels"] == 3)[0]
    # _, _, idxs = cleanup_boxes(target["boxes"][idxs_columns], target["scores"][idxs_columns], 0.25)
    # idxs_cleaned = idxs_columns[idxs]
    # idxs = torch.where(target["labels"] == 1)[0].tolist() + \
    #     torch.where(target["labels"] == 2)[0].tolist() + idxs_cleaned.tolist()
    #
    # for ii in idxs:
    #     box = target["boxes"][ii]
    #     plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
    #                                   facecolor="none",
    #                                   edgecolor=colors[target["labels"][ii].item()],
    #                                   alpha=0.5))
    #     if "scores" in target.keys():
    #         plt.text(box[0], box[1], round(target["scores"][ii].item(), 2),
    #                  color=colors[target["labels"][ii].item()])

    idxs = batch_cleanup_boxes(target["boxes"], target["scores"], target["labels"])

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
    idxs = torchvision.ops.nms(boxes, scores, threshold)
    cleaned_boxes = boxes[idxs]
    cleaned_scores = scores[idxs]
    return cleaned_boxes, cleaned_scores, idxs


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
