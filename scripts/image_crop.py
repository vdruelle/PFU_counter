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
    cx = original_box[0]
    cy = original_box[1]
    w = original_box[2]
    h = original_box[3]

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


def crop_lab_image(image_path, label_path):
    """
    Crops the image for all the labels and returns a list of cropped image. The first is the plate name, the
    second the phage names, and the rest are the dilution columns.
    """
    image = Image.open(image_path)
    image = image.transpose(Image.ROTATE_270)

    # Loading labels and putting them in in the coordinates of the 4 angles of the box
    df = pd.read_csv(label_path, sep=" ", names=["label", "cx", "cy", "w", "h"])
    cropped_images = []

    for index, row in df.iterrows():
        box = row[["cx", "cy", "w", "h"]]
        box = relative_to_absolute_box(box, image.size)

        # Cropping
        cropped_images = cropped_images + [crop_image(image, box)]

    return cropped_images


def make_column_images():
    """
    Create the phage dilution colmuns images from the orginal dataset of images and saves them in the
    data/phage_columns folder.
    """
    import os

    if not os.path.exists("data/phage_columns"):
        os.mkdir("data/phage_columns")

    raw_list = os.listdir("data/lab_raw_good/")
    for image_path in raw_list:
        image_path = "data/lab_raw_good/" + image_path
        name = image_path.split(".")[0]
        name = name.split("/")[2]
        label_path = "data/lab_raw_good_labels/" + name + ".txt"
        cropped_list = crop_lab_image(image_path, label_path)

        for idx, im in enumerate(cropped_list[2:]):
            im_name = "data/phage_columns/" + name + "_" + str(idx) + ".jpg"
            im.save(im_name)


def crop_column_image(image_path, label_path):
    """
    Crops the column image for all the labelled spots returns a list of cropped image. There are usually 1, 2
    or 3 elements.
    """
    image = Image.open(image_path)
    # image = image.transpose(Image.ROTATE_270) # Images are already rotated so no need for that

    # Loading labels and putting them in in the coordinates of the 4 angles of the box
    df = pd.read_csv(label_path, sep=" ", names=["label", "cx", "cy", "w", "h"])
    cropped_images = []

    for index, row in df.iterrows():
        box = row[["cx", "cy", "w", "h"]]
        box = relative_to_absolute_box(box, image.size)

        # Cropping
        cropped_images = cropped_images + [crop_image(image, box)]

    return cropped_images


def make_spot_images():
    """
    Create the phage dilution spot images from the column images and saves them in the data/phage_spots
    folder.
    """
    import os

    if not os.path.exists("data/phage_spots"):
        os.mkdir("data/phage_spots")

    raw_list = os.listdir("data/phage_columns/")
    for image_path in raw_list:
        image_path = "data/phage_columns/" + image_path
        name = image_path.split(".")[0]
        name = name.split("/")[2]
        label_path = "data/phage_columns_labels/" + name + ".txt"

        if os.path.exists(label_path):  # some columns don't have any good spots
            cropped_list = crop_column_image(image_path, label_path)

            for idx, im in enumerate(cropped_list):
                im_name = "data/phage_spots/" + name + "_" + str(idx) + ".jpg"
                im.save(im_name)


if __name__ == '__main__':
    # make_column_images()
    # make_spot_images()

    # image = Image.open("data/phage_columns/20200204_115031_0.jpg")
    # tmp = np.array(image)
    # plt.figure()
    # plt.imshow(np.rot90(tmp, axes=(0, 1)))
    # tmp = np.mean(tmp, axis=1)
    #
    # plt.figure()
    # plt.plot(tmp)
    # plt.show()

    import os
    m, mm = 0, 0
    for image_path in os.listdir("data/phage_spots/"):
        image = Image.open("data/phage_spots/" + image_path)
        image = np.array(image)
        m = max(image.shape[0], m)
        mm = max(image.shape[1], mm)
        print(image.shape[0]/8, image.shape[1]/8)

    print("max shape")
    print(m, mm)
