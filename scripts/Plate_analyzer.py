"""
This file contains the code necessary to run the pipeline for plate analysis starting from raw images to
the resulting dilutions
"""
import torch
import scipy
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

import utils
from model import PlateDetector, UNet


def plate_detection(image_path, plate_detector_save):
    """
    Loads image and plate_detector model form the given paths, computes the prediction of the plate detector
    and returns these predictions (in forms of torch tensor on GPU).
    """
    # Setting up the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    model = PlateDetector(num_classes=4, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load(plate_detector_save))
    model.eval()

    # Image loading
    image = utils.load_image_from_file(image_path)
    image = torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32)
    image.to(device)

    # Predictions
    image = image.to(device)
    with torch.no_grad():
        outputs = model([image])
    output = outputs[0]

    return image, output


def plate_extraction(image, detection):
    """
    Takes the image and the output of the detector (cleaned from overlapping boxes etc...), crop the subimages
    and returns them in a dictionary (just image for plate name and phage names, list of image for columns).
    """
    detector_images = {}
    sub_images = []
    for ii in range(detector_output_cleaned["labels"].shape[0]):
        box = torch.round(detector_output_cleaned["boxes"][ii]).type(torch.int32)
        sub_images += [image[:, box[1]:box[3], box[0]:box[2]]]

    detector_images["plate_name"] = sub_images[torch.where(detector_output_cleaned["labels"] == 1)[0]]
    detector_images["phage_names"] = sub_images[torch.where(detector_output_cleaned["labels"] == 2)[0]]
    detector_images["phage_spots"] = []

    for ii in torch.where(detector_output_cleaned["labels"] == 3)[0].tolist():
        detector_images["phage_spots"] += [sub_images[ii]]

    return detector_images

def map_spots(detector_output, spot_size, slack=0.7):
    """
    Computes the columns and rows of the dilution spots. Trivial way of doing it for now, might need some
    improvements.
    """
    spot_boxes = detector_output["boxes"][detector_output["labels"] == 3]
    avg_x = torch.mean(torch.stack((spot_boxes[:, 0], spot_boxes[:, 2])), axis=0)
    avg_y = torch.mean(torch.stack((spot_boxes[:, 1], spot_boxes[:, 3])), axis=0)

    rows = torch.zeros_like(avg_x)
    columns = torch.zeros_like(avg_x)

    counter = 1
    # Detect the columns
    for ii in range(columns.shape[0]):
        if columns[ii] == 0:
            x = avg_x[ii]
            similar = torch.logical_and((avg_x < (x + spot_size * slack)),
                                        (avg_x > (x - spot_size * slack)))
            columns[similar] = counter
            counter += 1

    # Detect the rows
    counter = 1
    for ii in range(rows.shape[0]):
        if rows[ii] == 0:
            y = avg_y[ii]
            similar = torch.logical_and((avg_y < (y + spot_size * slack)),
                                        (avg_y > (y - spot_size * slack)))
            rows[similar] = counter
            counter += 1

    return rows, columns


if __name__ == '__main__':
    plate_detector_save = "model_saves/Plate_detection.pt"
    phage_counter_save = "model_saves/Counter_phages.pt"
    image_path = "data/plates_labeled/spot_labeling/images/20200204_115135.jpg"
    # image_path = "data/plates_labeled/spot_labeling/images/20200204_115534.jpg"
    show_intermediate = False

    # --- Plate detection part ---
    image, detector_output = plate_detection(image_path, plate_detector_save)

    # --- Filtering of the detection ---
    detector_output_cleaned = utils.clean_plate_detector_output(detector_output, 0.15, 0.3)
    if show_intermediate:
        utils.plot_plate_detector(image, detector_output_cleaned)

    # --- Extraction of images from box detection ---
    detector_images = plate_extraction(image, detector_output_cleaned)

    # --- Sorting dilution spots into rows and columns  ---
    median_spot_size = np.median([[x.shape[1], x.shape[2]] for x in detector_images["phage_spots"]])
    rows, columns = map_spots(detector_output_cleaned, median_spot_size)

    plt.show()
