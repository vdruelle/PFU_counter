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
    image = np.transpose(image, (1, 2, 0))
    label = np.transpose(label, (1, 2, 0))

    plt.figure(figsize=(14, 10))
    plt.imshow(image)

    plt.figure(figsize=(14, 10))
    plt.imshow(label)
    plt.show()
