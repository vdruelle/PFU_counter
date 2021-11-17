"""
This file contains the code necessary to run the pipeline for plate analysis starting from raw images to
the resulting dilutions
"""
import torch
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

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
    image = image.to(device)

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
    boxes = []
    for ii in range(detection["labels"].shape[0]):
        box = torch.round(detection["boxes"][ii]).type(torch.int32)
        sub_images += [image[:, box[1]:box[3], box[0]:box[2]]]
        boxes += [box.cpu().tolist()]

    idx = torch.where(detection["labels"] == 1)[0]
    detector_images["plate_name"] = {"image": sub_images[idx], "bbox": boxes[idx]}
    idx = torch.where(detection["labels"] == 2)[0]
    detector_images["phage_names"] = {"image": sub_images[idx], "bbox": boxes[idx]}
    detector_images["phage_spots"] = []

    for ii in torch.where(detection["labels"] == 3)[0].tolist():
        detector_images["phage_spots"] += [{"image": sub_images[ii], "bbox": boxes[ii]}]

    return detector_images


def map_spots(detector_output, spot_size, slack=0.7):
    """
    Computes the columns and rows of the dilution spots. Trivial way of doing it for now, might need some
    improvements.
    Returns them in an ordered way starting at 1 (ie column 1 is first phage, row 1 is lowest dilution)
    """
    spot_boxes = detector_output["boxes"][detector_output["labels"] == 3]
    avg_x = torch.mean(torch.stack((spot_boxes[:, 0], spot_boxes[:, 2])), axis=0)
    avg_y = torch.mean(torch.stack((spot_boxes[:, 1], spot_boxes[:, 3])), axis=0)
    x_sorted = torch.sort(avg_x)
    y_sorted = torch.sort(avg_y, descending=True)  # Because low Y is top of image

    rows = torch.zeros_like(avg_x, dtype=torch.int)
    columns = torch.zeros_like(avg_x, dtype=torch.int)

    counter = 1
    # Detect the columns
    for ii in x_sorted.indices:
        if columns[ii] == 0:
            x = avg_x[ii]
            similar = torch.logical_and((avg_x < (x + spot_size * slack)),
                                        (avg_x > (x - spot_size * slack)))
            columns[similar] = counter
            counter += 1

    # Detect the rows
    counter = 1
    for ii in y_sorted.indices:
        if rows[ii] == 0:
            y = avg_y[ii]
            similar = torch.logical_and((avg_y < (y + spot_size * slack)),
                                        (avg_y > (y - spot_size * slack)))
            rows[similar] = counter
            counter += 1

    return rows, columns


def spot_selection(detector_images, columns, rows):
    """
    Select the dilution spots where we should count the colonies. Very trivial for now, might need to
    improve later.
    """
    for im in detector_images["phage_spots"]:
        im["to_count"] = False

    unique_columns = torch.unique(columns)
    for column in unique_columns:
        idxs = torch.where(columns == column)[0]
        spots_rows = []
        for idx in idxs:
            spots_rows += [detector_images["phage_spots"][idx]["row"]]
        spots_rows = torch.tensor(spots_rows)
        row1 = torch.max(spots_rows)
        spots_rows[torch.argmax(spots_rows)] = -1
        row2 = torch.max(spots_rows)

        # Very ugly way to tag the images but should do the trick
        for idx in idxs:
            if detector_images["phage_spots"][idx]["column"] == column:
                if detector_images["phage_spots"][idx]["row"] == row1:
                    detector_images["phage_spots"][idx]["to_count"] = True
                elif detector_images["phage_spots"][idx]["row"] == row2:
                    detector_images["phage_spots"][idx]["to_count"] = True

    return detector_images


def count_to_concentration(counts, row):
    """
    Computes concentration according to number of counts and dilution row of the spot.
    """
    return counts * 10**row


def count_spots_density(detector_images, counter_save, scaling=1000):
    """
    Uses the colony counting network to predict the count in each spot to count. Returns the
    detector_images dict with additional keys for the counts in the spot images.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    model = UNet()
    model.to(device)
    model.load_state_dict(torch.load(counter_save))
    model.eval()
    with torch.no_grad():
        for image_dict in detector_images["phage_spots"]:
            if image_dict["to_count"]:
                tmp = image_dict["image"].cpu().numpy().transpose(1, 2, 0)
                tmp = utils.pad_image_to_correct_size(tmp)
                tmp = torch.tensor(tmp.transpose(2, 0, 1)).to(device)
                tmp = torch.unsqueeze(tmp, 0)  # adding one dimension
                output = model(tmp)
                output[output < 0] = 0

                nb_predicted = torch.sum(output).item() / scaling
                image_dict["counts"] = nb_predicted

    for image_dict in detector_images["phage_spots"]:
        if image_dict["to_count"]:
            image_dict["concentration"] = count_to_concentration(image_dict["counts"], image_dict["row"])

    return detector_images


def count_spots_box(detector_images, counter_save, score_threshold=0.1):
    """
    Uses the colony counting network to predict the count in each spot to count. Returns the
    detector_images dict with additional keys for the counts in the spot images.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    model = PlateDetector(num_classes=2, backbone="mobilenet", trainable_backbone_layers=None)
    model.to(device)
    model.load_state_dict(torch.load(counter_save))
    model.eval()
    with torch.no_grad():
        for image_dict in detector_images["phage_spots"]:
            if image_dict["to_count"]:
                output = model(image_dict["image"].unsqueeze(0))[0]  # adding 1 dimension
                image_dict["counts"] = output["scores"][output["scores"] > score_threshold].shape[0]

    for image_dict in detector_images["phage_spots"]:
        if image_dict["to_count"]:
            image_dict["concentration"] = count_to_concentration(image_dict["counts"], image_dict["row"])

    return detector_images


def make_analysis_output(detector_images):
    """
    Computes the analysis output from the detector_images dictionary.
    """
    # Saving the plate and phage names as numpy arrays
    plate_name = np.transpose(detector_images["plate_name"]["image"].cpu().numpy(), (1, 2, 0))
    phage_names = np.transpose(detector_images["phage_names"]["image"].cpu().numpy(), (1, 2, 0))
    analysis_output = {"plate_name": plate_name, "phage_names": phage_names}

    spots = detector_images["phage_spots"]
    columns = [im["column"] for im in spots if im["to_count"]]
    columns = np.unique(columns)

    for col in columns:
        analysis_output[col] = [s["concentration"]
                                for s in spots if (s["column"] == col and s["to_count"])]
    return analysis_output


if __name__ == '__main__':
    plate_detector_save = "model_saves/Plate_detector2.pt"
    phage_counter_save = "model_saves/Colony_counter.pt"
    # image_path = "data/plates_labeled/images/20200204_115135.jpg"
    # image_path = "data/plates_labeled/images/20200204_115534.jpg"
    image_path = "data/plates_labeled/test/images/20211112_104047.jpg"
    show = True

    # --- Plate detection part ---
    image, detector_output = plate_detection(image_path, plate_detector_save)

    # --- Filtering of the detection ---
    detector_output_cleaned = utils.clean_plate_detector_output(detector_output, 0.15, 0.3)

    # --- Extraction of images from box detection ---
    detector_images = plate_extraction(image, detector_output_cleaned)

    # --- Sorting dilution spots into rows and columns  ---
    median_spot_size = np.median([[x["image"].shape[1], x["image"].shape[2]]
                                  for x in detector_images["phage_spots"]])
    rows, columns = map_spots(detector_output_cleaned, median_spot_size)
    for ii, spot in enumerate(detector_images["phage_spots"]):
        spot["row"] = rows[ii].item()
        spot["column"] = columns[ii].item()

    # --- Selecting dilution spots to count ---
    detector_images = spot_selection(detector_images, columns, rows)

    # --- Feeding to the colony counter network ---
    detector_images = count_spots_box(detector_images, phage_counter_save, 0.3)

    # --- Image feedback ---
    if show:
        utils.plot_plate_analysis(image, detector_images)

    # --- Plate_analyzer output ---
    analysis_output = make_analysis_output(detector_images)
    plt.show()
