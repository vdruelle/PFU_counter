"""A script with helper function to manage the data used by the PFU_counter."""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import pandas as pd
import random
import torch


def inspect_plate_data(image_label_folder="data/plates_labeled/", start_idx=0):
    """
    Shows the plate images and respective labels in the folder. Use the start_idx argument to specify at which
    image of the list you wish to start.
    """
    import utils
    import matplotlib.pyplot as plt
    img_size = (4608, 3456)

    image_list = list(sorted(os.listdir(image_label_folder + "images/")))
    labels_list = list(sorted(os.listdir(image_label_folder + "labels/")))
    for image_name, label_name in zip(image_list[start_idx:], labels_list[start_idx:]):
        print(f"Image : {image_name}")
        image = utils.load_image_from_file(image_label_folder + "images/" + image_name, dtype="int")

        boxes, labels = utils.boxes_and_labels_from_file(
            image_label_folder + "labels/" + label_name, *img_size)

        target = {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)}
        utils.plot_plate_detector(torch.tensor(np.transpose(image, (2, 0, 1))), target)
        plt.show()


def convert_heic_to_jpg(image_folder):
    """
    Converts all the images in a folder from .heic to .jpg.
    """
    image_list = list(sorted(os.listdir(image_folder)))
    for image_name in image_list:
        os.system(f"heif-convert {image_folder + image_name} {image_folder+image_name[:-4]+'jpg'} -q 100")


def remove_orientation(folder_path, destination_folder):
    """
    Process the image in the define folder and adds them to the destination_folder.
    """
    from PIL import Image
    os.makedirs(destination_folder, exist_ok=True)
    image_list = list(sorted(os.listdir(folder_path)))

    for image_name in image_list:
        os.system(f"exiftool -Orientation= {folder_path+image_name} -o {destination_folder+image_name}")
        image = Image.open(destination_folder + image_name)
        image = image.transpose(Image.ROTATE_270)
        image.save(destination_folder + image_name)


def create_plate_data(image_label_folder="data/plates_labeled/", valid_size=12):
    """
    Creates the plate datasets (train + test) and saves them in the corresponding folder.
    """
    image_list = list(sorted(os.listdir(os.path.join(image_label_folder, "images"))))
    os.makedirs(image_label_folder + "train/images/", exist_ok=True)
    os.makedirs(image_label_folder + "train/labels/", exist_ok=True)
    os.makedirs(image_label_folder + "test/images/", exist_ok=True)
    os.makedirs(image_label_folder + "test/labels/", exist_ok=True)

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


def make_spots_label(path_csv, save_folder):
    """
    Creates the image labels for the phage spots from the csv label file and save them in the save_folder.
    """
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


def kdtree_gaussian(gt):
    """
    Gaussian smoothing to predict density target for colony counter. This ones sizes the gaussians based on
    the distance to the closest colonies.
    """
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)
    d_median = np.median(distances[:, 1:])  # removing first column because its distance to same point, i.e. 0

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            d_neighbors = distances[i][1] + distances[i][2] + distances[i][3]
            sigma = min(2 * d_median, d_neighbors) * 0.1
        else:
            print("There is only one point in the label")
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density


def smooth_spots_label(raw_folder, output_folder, mode="kdtree"):
    """
    Takes the raw images label, and smooth them using a gaussian kernel to get the target density map. Saves
    the density map as numpy arrays (because they are floats).
    """

    assert mode in ["kdtree", "standard"], "Mode must be 'kdtree' or 'standard'"

    import utils
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    raw_list = os.listdir(raw_folder)
    if "labels.csv" in raw_list:
        raw_list.remove("labels.csv")
    for raw in raw_list:
        raw_label = utils.load_image_from_file(raw_folder + raw, dtype="int")
        if len(raw_label.shape) == 3:  # specific to the cell dataset, where the labels are RGB
            raw_label = np.sum(raw_label, axis=2)
        if mode == "kdtree":
            density = kdtree_gaussian(raw_label)
        elif mode == "standard":
            density = gaussian_filter(raw_label / 255, sigma=(1, 1), order=0)
        density = density.astype(np.float32)
        np.save(output_folder + raw[:-4] + ".npy", density)


def inspect_spot_data(image_folder, density_folder):
    """
    Plots all the images and their density.
    """
    import utils
    import matplotlib.pyplot as plt

    images = list(sorted(os.listdir(image_folder)))
    densities = list(sorted(os.listdir(density_folder)))

    for im, de in zip(images, densities):
        image = utils.load_image_from_file(image_folder + im)
        density = np.load(density_folder + de)

        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        axs[0].imshow(image, interpolation=None)
        axs[1].imshow(density, interpolation=None)
        axs[1].set_xlabel(f"{round(np.sum(density))}")
        plt.show()


def make_box_density(image_folder, label_folder, output_folder, spread_scaling=50):
    """
    Creates the density from the box labels.
    """
    import utils
    labels = list(sorted(os.listdir(label_folder)))
    images = list(sorted(os.listdir(image_folder)))

    for image_name, label_name in zip(images, labels):
        label_path = label_folder + label_name
        image = utils.load_image_from_file(image_folder + image_name)
        density = np.zeros_like(image)[:, :, 0].astype(np.float32)
        df = pd.read_csv(label_path, sep=" ", names=["label", "cx", "cy", "w", "h"])
        for ii in range(len(df)):
            x = round(df["cx"][ii] * image.shape[1])
            y = round(df["cy"][ii] * image.shape[0])
            sigma = np.array([df["w"][ii], df["h"][ii]]) * spread_scaling
            tmp = np.zeros_like(image)[:, :, 0].astype(float)
            tmp[y, x] = 1
            density += gaussian_filter(tmp, sigma=sigma, order=0)
        np.save(output_folder + image_name[:-4] + ".npy", density)


if __name__ == '__main__':
    # inspect_plate_data("data/plates_labeled/", start_idx=0)
    # convert_heic_to_jpg(image_folder="data/plates_raw/11-11-2021/heic/")
    # remove_orientation("data/plates_raw/11-11-2021/png/", "data/plates_raw/11-11-2021_oriented/")
    # create_plate_data()
    # add_plate_data("data/plates_raw/square_10-11-2021/", "data/plates_raw/square_10-11-2021_oriented/")
    # make_spots_label("data/phage_spots_minimal/dot_labeling/test/labels/labels.csv",
    #                  "data/phage_spots_minimal/dot_labeling/test/labels/")
    # smooth_spots_label("data/phage_spots_subset/test/labels/",
    #                    "data/phage_spots_subset/test/density_kdtree/", mode="kdtree")
    # inspect_spot_data("data/phage_spots_minimal/box_labeling/train/images/",
    #                   "data/phage_spots_minimal/box_labeling/train/density/")
    # density = make_box_density("data/phage_spots_minimal/box_labeling/test/images/",
    #                            "data/phage_spots_minimal/box_labeling/test/labels/",
    #                            "data/phage_spots_minimal/box_labeling/test/density/")
