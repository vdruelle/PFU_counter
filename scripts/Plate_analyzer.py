"""
This file contains the code necessary to run the pipeline for plate analysis starting from raw images to
the resulting dilutions
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

import utils
from model import PlateDetector


def plate_detection(image_path, plate_detector_save):
    """
    Loads image and plate_detector model form the given paths, computes the prediction of the plate detector
    and returns these predictions.
    """
    # Setting up the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else print("GPU not available"))
    model = PlateDetector()
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


if __name__ == '__main__':
    plate_detector_save = "model_saves/Plate_detection.pt"
    phage_counter_save = "model_saves/Counter_phages.pt"
    image_path = "data/lab_raw_good/20200204_115135.jpg"
    show_intermediate = True

    # --- Plate detection part ---
    image, detector_output = plate_detection(image_path, plate_detector_save)
    idxs = utils.batch_cleanup_boxes(
        detector_output["boxes"], detector_output["scores"], detector_output["labels"], 0.15)
    detector_output_cleaned = {k: v[idxs] for k, v in detector_output.items()}
    if show_intermediate:
        utils.plot_plate_detector(image, detector_output)

    # --- Extraction of images from box detection ---
    detector_images = {}
    sub_images = []
    for ii in range(detector_output_cleaned["labels"].shape[0]):
        box = torch.round(detector_output_cleaned["boxes"][ii]).type(torch.int32)
        sub_images += [image[:, box[1]:box[3], box[0]:box[2]]]

    detector_images["plate_name"] = sub_images[torch.where(detector_output_cleaned["labels"] == 1)[0]]
    detector_images["phage_names"] = sub_images[torch.where(detector_output_cleaned["labels"] == 2)[0]]
    detector_images["phage_columns"] = []
    for ii in torch.where(detector_output_cleaned["labels"] == 3)[0].tolist():
        detector_images["phage_columns"] += [sub_images[ii]]

    # for image in detector_images["phage_columns"]:
    #     image = image.cpu().numpy().transpose((1, 2, 0))
    #     plt.figure()
    #     plt.imshow(image)
