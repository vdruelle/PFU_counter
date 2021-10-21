"""A tool to download and preprocess data, and generate HDF5 file."""
import os
import shutil
import zipfile
from glob import glob
import wget
from typing import List, Tuple

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import pandas as pd
import random


def create_hdf5(dataset_name: str,
                train_size: int,
                valid_size: int,
                img_size: Tuple[int, int],
                in_channels: int = 3):
    """
    Create empty training and validation HDF5 files with placeholders
    for images and labels (density maps).
    Note:
    Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
    Existing files will be overwritten.
    Args:
        dataset_name: used to create a folder for train.h5 and valid.h5
        train_size: no. of training samples
        valid_size: no. of validation samples
        img_size: (width, height) of a single image / density map
        in_channels: no. of channels of an input image
    Returns:
        A tuple of pointers to training and validation HDF5 files.
    """
    # create output folder if it does not exist
    os.makedirs(dataset_name, exist_ok=True)

    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(os.path.join(dataset_name, 'train.h5'), 'w')
    valid_h5 = h5py.File(os.path.join(dataset_name, 'valid.h5'), 'w')

    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, *img_size, in_channels))
        h5.create_dataset('labels', (size, *img_size, 1))

    return train_h5, valid_h5


def get_and_unzip(url: str, location: str = "."):
    """Extract a ZIP archive from given URL.
    Args:
        url: url of a ZIP file
        location: target location to extract archive in
    """
    dataset = wget.download(url)
    dataset = zipfile.ZipFile(dataset)
    dataset.extractall(location)
    dataset.close()
    os.remove(dataset.filename)


def generate_cell_data():
    """Generate HDF5 files for fluorescent cell dataset."""
    # download and extract dataset
    get_and_unzip(
        'http://www.robots.ox.ac.uk/~vgg/research/counting/cells.zip',
        location='data/cells'
    )
    # create training and validation HDF5 files
    train_h5, valid_h5 = create_hdf5('data/cell',
                                     train_size=150,
                                     valid_size=50,
                                     img_size=(256, 256),
                                     in_channels=3)

    # get the list of all samples
    # dataset name convention: XXXcell.png (image) XXXdots.png (label)
    image_list = glob(os.path.join('data/cells', '*cell.*'))
    image_list.sort()

    def fill_h5(h5, images):
        """
        Save images and labels in given HDF5 file.
        Args:
            h5: HDF5 file
            images: the list of images paths
        """
        for i, img_path in enumerate(images):
            # get label path
            label_path = img_path.replace('cell.png', 'dots.png')
            # get an image as numpy array
            image = np.array(Image.open(img_path), dtype=np.float32) / 255

            # convert a label image into a density map: dataset provides labels
            # in the form on an image with red dots placed in objects position

            # load an RGB image
            label = np.array(Image.open(label_path))
            # make a one-channel label array with 100 in red dots positions
            label = 100.0 * (label[:, :, 0] > 0)
            # generate a density map by applying a Gaussian filter
            label = gaussian_filter(label, sigma=(1, 1), order=0)
            label = np.expand_dims(label, axis=-1)

            # save data to HDF5 file
            h5['images'][i] = image
            h5['labels'][i] = label

    # use first 150 samples for training and the last 50 for validation
    fill_h5(train_h5, image_list[:150])
    fill_h5(valid_h5, image_list[150:])

    # close HDF5 files
    train_h5.close()
    valid_h5.close()

    # cleanup
    # shutil.rmtree('data/cells')


def make_spots_label(path_csv, save_folder):
    """
    Creates the image labels for the phage spots from the csv label file and save them in the save_folder.
    """
    import pandas as pd
    df = pd.read_csv(path_csv, sep=",", names=["label", "x", "y", "image", "w", "h"])
    os.makedirs(save_folder, exist_ok=True)

    for image_name in np.unique(df["image"]):
        tmp = df[df["image"] == image_name]

        label_image = np.zeros(shape=(tmp["w"].values[0], tmp["h"].values[0]))
        label_image[tmp["x"], tmp["y"]] = 255
        label_image = Image.fromarray(label_image.transpose()).convert("L")
        label_image.save(save_folder + image_name.split(".")[0] + "_labels.png")


def resize_spot_label(label, img_size=(256, 256), value=100):
    """
    Resize the spot label image to the desired shape.
    """
    label = np.array(label)
    # Moving the label dot by hand because resize from PIL changes the sum value of the labels
    x, y = np.where(label > 0)
    x = np.round(x * (img_size[0] / label.shape[0])).astype(int)
    y = np.round(y * (img_size[1] / label.shape[1])).astype(int)
    label = np.zeros(img_size)
    label[x, y] = value
    return label


def generate_phage_data():
    """
    Generates the h5 files for the phage colonies dataset.
    """
    dataset_name = "data/phage_colonies/"
    spots_image_folder = "data/phage_spots/"
    spots_label_folder = "data/phage_spots_labels/"
    os.makedirs(dataset_name, exist_ok=True)

    # Defining length of the train and valid dataset
    nb_labeled = len(os.listdir(spots_label_folder)) - 1  # -1 for the CSV file in there
    valid_size = 20
    train_size = nb_labeled - valid_size
    img_size = (256, 256)
    label_list = os.listdir(spots_label_folder)
    label_list.remove("labels.csv")

    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(os.path.join(dataset_name, 'train.h5'), 'w')
    valid_h5 = h5py.File(os.path.join(dataset_name, 'valid.h5'), 'w')
    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, *img_size, 3))
        h5.create_dataset('labels', (size, *img_size, 1))

    def fill_h5_spots(h5, labels):
        """
        Save images and labels in given HDF5 file.
        Args:
            h5: HDF5 file
            images: the list of label images paths
        """
        for ii, label_name in enumerate(labels):  # Only images that have labels
            image_name = label_name.replace("_labels", "")

            image = Image.open(spots_image_folder + image_name)
            image = image.resize(img_size)
            image = np.array(image) / 255

            label = Image.open(spots_label_folder + label_name)
            label = resize_spot_label(label, img_size, 100)
            # generate a density map by applying a Gaussian filter
            label = gaussian_filter(label, sigma=(1, 1), order=0)
            label = np.expand_dims(label, axis=-1)

            # save data to HDF5 file
            h5['images'][ii] = image
            h5['labels'][ii] = label

    # Split between train and validation
    fill_h5_spots(train_h5, label_list[:train_size])
    fill_h5_spots(valid_h5, label_list[train_size:])

    train_h5.close()
    valid_h5.close()


def generate_plate_data():
    """
    Generates the h5 files for the phage colonies dataset.
    """
    dataset_name = "data/phage_plates/"
    image_folder = "data/lab_raw_good/"
    label_folder = "data/lab_raw_good_labels/"
    os.makedirs(dataset_name, exist_ok=True)

    # Shape of the orinial images
    img_size = (4608, 3456)
    box_size = (20, 4)  # at maximum 20 boxes per images
    label_size = 20  # at maximum 20 labels per images, corresponding to the 20 boxes
    image_list = list(sorted(os.listdir(image_folder)))
    valid_size = 6
    train_size = len(image_list) - valid_size
    valid_list = random.choices(image_list, k=valid_size)
    train_list = [im for im in image_list if im not in valid_list]

    # create HDF5 files: [dataset_name]/(train | valid).h5
    train_h5 = h5py.File(os.path.join(dataset_name, 'train.h5'), 'w')
    valid_h5 = h5py.File(os.path.join(dataset_name, 'valid.h5'), 'w')
    # add two HDF5 datasets (images and labels) for each HDF5 file
    for h5, size in ((train_h5, train_size), (valid_h5, valid_size)):
        h5.create_dataset('images', (size, *img_size, 3))
        h5.create_dataset('boxes', (size, *box_size))
        h5.create_dataset('labels', (size, label_size))

    def fill_h5_plate(h5, images):
        """
        Save images and labels in given HDF5 file.
        Args:
            h5: HDF5 file
            images: the list of label images paths
        """
        for ii, image_name in enumerate(images):
            label_name = image_name.replace(".jpg", ".txt")  # images and label have same name up to extension

            image = Image.open(image_folder + image_name)
            image = image.transpose(Image.ROTATE_270)  # Because PIL consider longest side to be width
            image = np.array(image) / 255

            boxes, labels = boxes_from_label_file(label_folder + label_name, *img_size, max_length=label_size)

            # save data to HDF5 file
            h5['images'][ii] = image
            h5['boxes'][ii] = np.array(boxes, dtype=np.float32)
            h5['labels'][ii] = np.array(labels, dtype=np.int64)

    # Split between train and validation
    fill_h5_plate(train_h5, train_list)
    fill_h5_plate(valid_h5, valid_list)

    train_h5.close()
    valid_h5.close()


def boxes_from_label_file(label_file, image_height, image_width, max_length, pad_value=-1):
    """
    Returns labels and boxes as lists, as expected by the RCNN network. Pad with -1 to max length to get
    constant output sizes.
    """
    df = pd.read_csv(label_file, sep=" ", names=["label", "cx", "cy", "w", "h"])
    df["label"] += 1  # label 0 must be background
    df2 = df.copy(deep=True)
    df2.columns = ["label", "x1", "y1", "x2", "y2"]
    df2["x1"] = (df["cx"] - df["w"] / 2.0) * image_width
    df2["y1"] = (df["cy"] - df["h"] / 2.0) * image_height
    df2["x2"] = (df["cx"] + df["w"] / 2.0) * image_width
    df2["y2"] = (df["cy"] + df["h"] / 2.0) * image_height
    boxes = df2[["x1", "y1", "x2", "y2"]].values.tolist()
    boxes += [[pad_value, pad_value, pad_value, pad_value]] * (max_length - len(boxes))  # padding
    labels = df["label"].values.tolist()
    labels += [pad_value] * (max_length - len(labels))  # padding

    return boxes, labels


def inspect_plate_data(image_label_folder="data/plates_labeled/", start_idx=0):
    """
    Shows the plate images and respective labels in the folder. Use the start_idx argument to specify at which
    image of the list you wish to start.
    """
    import utils
    import matplotlib.pyplot as plt
    img_size = (4608, 3456)
    label_size = 20

    image_list = list(sorted(os.listdir(image_label_folder + "images/")))
    labels_list = list(sorted(os.listdir(image_label_folder + "labels/")))
    for image_name, label_name in zip(image_list[start_idx:], labels_list[start_idx:]):
        print(f"Image : {image_name}")
        image = utils.load_image_from_file(image_label_folder + "images/" + image_name, dtype="int")

        boxes, labels = boxes_from_label_file(
            image_label_folder + "labels/" + label_name, *img_size, max_length=label_size)

        utils.plot_plate_data(image, boxes, labels)
        plt.show()


def create_plate_data(image_label_folder="data/plates_labeled/"):
    """
    Creates the plate datasets (train + test) in numpy/list format and saves them in the corresponding folder.
    """
    image_list = list(sorted(os.listdir(os.path.join(image_label_folder, "images"))))
    os.makedirs(image_label_folder + "train/images/", exist_ok=True)
    os.makedirs(image_label_folder + "train/labels/", exist_ok=True)
    os.makedirs(image_label_folder + "test/images/", exist_ok=True)
    os.makedirs(image_label_folder + "test/labels/", exist_ok=True)

    valid_size = 12
    valid_list = random.choices(image_list, k=valid_size)
    train_list = [im for im in image_list if im not in valid_list]

    # Train data
    for ii in range(len(train_list)):
        old_image_path = os.path.join(image_label_folder + "images/", train_list[ii])
        old_label_path = os.path.join(image_label_folder + "labels/", train_list[ii][:-4] + ".txt")

        new_image_path = os.path.join(image_label_folder + "train/images/", train_list[ii])
        new_label_path = os.path.join(image_label_folder + "train/labels/", train_list[ii][:-4] + ".txt")

        os.system(f"cp {old_image_path} {new_image_path}")
        os.system(f"cp {old_label_path} {new_label_path}")

    # Test data
    for ii in range(len(valid_list)):
        old_image_path = os.path.join(image_label_folder + "images/", valid_list[ii])
        old_label_path = os.path.join(image_label_folder + "labels/", valid_list[ii][:-4] + ".txt")

        new_image_path = os.path.join(image_label_folder + "test/images/", valid_list[ii])
        new_label_path = os.path.join(image_label_folder + "test/labels/", valid_list[ii][:-4] + ".txt")

        os.system(f"cp {old_image_path} {new_image_path}")
        os.system(f"cp {old_label_path} {new_label_path}")


if __name__ == '__main__':
    # generate_cell_data()
    # make_spots_label("data/phage_spots_labels/labels.csv", "data/phage_spots_labels/")
    # generate_phage_data()
    # generate_plate_data()
    # inspect_plate_data(start_idx=50)
    create_plate_data()
