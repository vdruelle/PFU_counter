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
    show_intermediate = False

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
    column_width = []
    for ii in torch.where(detector_output_cleaned["labels"] == 3)[0].tolist():
        detector_images["phage_columns"] += [sub_images[ii]]
        column_width += [sub_images[ii].shape[2]]
    mean_column_width = np.mean(column_width)

    # --- Extraction of spot from phage columns according to grayscale histogram
    for image in detector_images["phage_columns"]:
        # image = detector_images["phage_columns"][0]
        image = image.cpu().numpy().transpose((2, 1, 0))
        image = np.sum(image, axis=2)
        # plt.figure()
        # plt.imshow(image, cmap="gray")
        image = scipy.ndimage.gaussian_filter(image, 25)
        nb_dilution_from_shape = image.shape[1] / image.shape[0]

        plt.figure()
        hist = np.sum(image, axis=0)
        plt.plot(hist)
        min_peak_distance = 0.7 * mean_column_width
        peak_idxs = scipy.signal.find_peaks(hist, distance=min_peak_distance)[0]
        valley_idxs = scipy.signal.find_peaks(-hist, distance=min_peak_distance)[0]
        plt.plot(peak_idxs, hist[peak_idxs], 'r.')
        plt.plot(valley_idxs, hist[valley_idxs], 'g.')

        # Selection of the size of plaques
        peak_diff = np.diff(peak_idxs)
        valley_diff = np.diff(valley_idxs)
        peak_to_valley = 2 * np.diff(np.sort(np.concatenate((peak_idxs, valley_idxs))))
        all_diffs = np.concatenate((peak_diff, valley_diff, peak_to_valley))
        inferred_spot_size = round(np.median(all_diffs))
        if inferred_spot_size > 1.2 * mean_column_width or inferred_spot_size < 0.7 * mean_column_width:
            print("There might be an error in spot size estimate." +
                  f" \nInferred spot size is {inferred_spot_size} while column width is {mean_column_width}")

        size_threshold = 0.1
        peaks_mask = np.logical_and(peak_diff < (1 + size_threshold) * inferred_spot_size,
                                    peak_diff > (1 - size_threshold) * inferred_spot_size)
        peaks_mask = np.logical_or(np.concatenate(
            (peaks_mask, [False])), np.concatenate(([False], peaks_mask)))
        good_peaks = peak_idxs[peaks_mask]
        plt.plot(good_peaks, hist[good_peaks], 'r.', markersize=10)

        valleys_mask = np.logical_and(valley_diff < (1 + size_threshold) * inferred_spot_size,
                                      valley_diff > (1 - size_threshold) * inferred_spot_size)
        valleys_mask = np.logical_or(np.concatenate(
            (valleys_mask, [False])), np.concatenate(([False], valleys_mask)))
        good_valleys = valley_idxs[valleys_mask]
        plt.plot(good_valleys, hist[good_valleys], 'g.', markersize=10)

        separations = np.copy(good_valleys)
        pred = 1
        while pred > 0:
            ref = separations[0]
            lower_peaks = good_peaks[good_peaks < ref]
            if lower_peaks.size > 0:
                lower_peak = lower_peaks[-1]
                pred = ref - 2 * (ref - lower_peak)
            else:
                pred = ref - inferred_spot_size
            separations = np.concatenate(([pred], separations))

        pred = 1
        while pred < image.shape[1]:
            ref = separations[-1]
            higher_peaks = good_peaks[good_peaks > ref]
            if higher_peaks.size > 0:
                higher_peak = higher_peaks[0]
                pred = ref + 2 * (higher_peak - ref)
            else:
                pred = ref + inferred_spot_size
            separations = np.concatenate((separations, [pred]))

        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.plot(separations, np.zeros_like(separations), 'g.')
