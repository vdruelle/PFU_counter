"""Function to crop the raw images into smaller parts"""
import torch
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
            im_name = "data/phage_columns/" + name + "_" + str(idx) + ".png"
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
                im_name = "data/phage_spots/" + name + "_" + str(idx) + ".png"
                im.save(im_name)


def crop_torch_image(image, box):
    """
    Returns a crop of the image defined by the box. Box is in format [x1, y1, x2, y2].
    """
    box = np.round(box).astype(int)
    return image[:, box[1]:box[3], box[0]:box[2]]


def fake_plate_analyzer_selection(image, boxes, labels):
    """
    Reproduces the selection of spots by the Plate_analyzer.
    """
    from Plate_analyzer import plate_extraction, map_spots, spot_selection

    boxes, labels = torch.tensor(boxes), torch.tensor(labels)

    detection = {"labels": labels, "boxes": boxes}
    detector_images = plate_extraction(image, detection)
    median_spot_size = np.median([[x.shape[1], x.shape[2]] for x in detector_images["phage_spots"]])
    rows, columns = map_spots(detection, median_spot_size)
    tmp = []
    for ii, spot in enumerate(detector_images["phage_spots"]):
        tmp += [{"image": spot, "row": rows[ii].item(), "column": columns[ii].item()}]
    detector_images["phage_spots"] = tmp

    # --- Selecting dilution spots to count ---
    detector_images = spot_selection(detector_images, columns, rows)
    return detector_images


def spots_from_labels(data_folder):
    """
    Creates all the spots image from the plate labels based on the plate_analyzer selection rule for which
    one should be used to count.
    """
    import os
    import utils

    if not os.path.exists("data/phage_spots"):
        os.mkdir("data/phage_spots")
    if not os.path.exists("data/phage_spots/images"):
        os.mkdir("data/phage_spots/images")

    image_list = os.listdir(data_folder + "images/")

    for image_path in image_list:
        image = utils.load_image_from_file(data_folder + "images/" + image_path, dtype="int")
        image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
        labels_path = image_path.split(".")[0] + ".txt"
        boxes, labels = utils.boxes_and_labels_from_file(
            data_folder + "labels/" + labels_path, image.shape[1], image.shape[2])

        spot_images = fake_plate_analyzer_selection(image, boxes, labels)["phage_spots"]
        spot_images = [spot for spot in spot_images if spot["to_count"]]

        for ii in range(len(spot_images)):
            im = np.array(spot_images[ii]["image"]).astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            Image.fromarray(im).save(f"data/phage_spots/images/{image_path[:-4]}_{ii}.jpg")


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

    # import os
    # m, mm = 0, 0
    # for image_path in os.listdir("data/phage_spots/"):
    #     image = Image.open("data/phage_spots/" + image_path)
    #     image = np.array(image)
    #     m = max(image.shape[0], m)
    #     mm = max(image.shape[1], mm)
    #     print(image.shape[0]/8, image.shape[1]/8)
    #
    # print("max shape")
    # print(m, mm)

    spots_from_labels("data/plates_labeled/spot_labeling/")
