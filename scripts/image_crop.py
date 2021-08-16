"""Function to crop the raw images into smaller parts"""
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def relative_to_absolute_box(original_box, image_size):
    """
    Takes the original box defined as [centerX, centerY, width, height] in relative coordinates and returns
    it in [left, upper, right, lower] if absolute pixel value.
    image_size is the tuple returned by PIL.Image.size.
    """
    image_width, image_height = image_size
    cx = box[0]
    cy = box[1]
    w = box[2]
    h = box[3]

    left = (cx - w / 2) * image_width
    upper = (cy - h / 2) * image_height
    right = (cx + w / 2) * image_width
    lower = (cy + h / 2) * image_height

    return [left, upper, right, lower]


def crop_image(image, box):
    """
    Crops the image in the position defined by the box and returns it.
    The box is the position of the [left, upper, right, lower] sides
    """
    cropped = image.crop(box)
    return cropped


if __name__ == '__main__':
    # Defining paths for test
    image_path = "data/lab_raw_good/20200204_115534.jpg"
    label_path = "data/lab_raw_good_labels/20200204_115534.txt"

    # Loading image
    image = Image.open(image_path)
    image = image.transpose(Image.ROTATE_270)

    # Loading labels and putting them in in the coordinates of the 4 angles of the box
    df = pd.read_csv(label_path, sep=" ", names=["label", "cx", "cy", "w", "h"])
    box = df[["cx", "cy", "w", "h"]].iloc[2].values
    box = relative_to_absolute_box(box, image.size)

    # Cropping
    cropped = crop_image(image, box)

    # Plot
    plt.figure()
    plt.imshow(image)

    plt.figure()
    plt.imshow(cropped)

    plt.show()
