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
    and returns these predictions (in forms of torch tensor on GPU).
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
    detector_images["phage_columns"] = []

    for ii in torch.where(detector_output_cleaned["labels"] == 3)[0].tolist():
        detector_images["phage_columns"] += [sub_images[ii]]

    return detector_images


def find_raw_peaks_and_valleys(hist, min_peak_to_peak_distance):
    """
    Finds the raw peaks and valleys from the Y histograms of the images and returns them.
    """
    peak_idxs = scipy.signal.find_peaks(hist, distance=min_peak_distance)[0]
    valley_idxs = scipy.signal.find_peaks(-hist, distance=min_peak_distance)[0]
    return peak_idxs, valley_idxs


def estimate_spot_size(peak_idxs, valley_idxs, column_width):
    """
    Find the most likely distance between spots from the peaks and valleys and returns it.
    """
    peak_diff = np.diff(peak_idxs)
    valley_diff = np.diff(valley_idxs)
    peak_to_valley = 2 * np.diff(np.sort(np.concatenate((peak_idxs, valley_idxs))))
    all_diffs = np.concatenate((peak_diff, valley_diff, peak_to_valley))
    inferred_spot_size = round(np.median(all_diffs))
    if inferred_spot_size > 1.2 * column_width or inferred_spot_size < 0.7 * column_width:
        print("There might be an error in spot size estimate." +
              f" \nInferred spot size is {inferred_spot_size} while column width is {column_width}")
    return inferred_spot_size


# def clean_peaks(idxs, inferred_spot_size, size_threshold=0.1):
#     """
#     Clean the valleys/peaks based on whether they are distant from other peaks/valleys by the inferred spot
#     size.
#     """
#     diff = np.diff(idxs)
#     mask = np.logical_and(diff < (1 + size_threshold) * inferred_spot_size,
#                           diff > (1 - size_threshold) * inferred_spot_size)
#     mask1 = np.concatenate((mask, [False]))
#     mask2 = np.concatenate(([False], mask))
#     mask = np.logical_or(mask1, mask2)
#     breakpoint()
#     good_idxs = idxs[mask]
#     return good_idxs


def best_peak_and_valley(peak_idxs, valley_idxs, inferred_spot_size, size_threshold=0.1):
    """
    Returns the best peak and best valley. How good a peak or valley is is decided based on the number of
    neighbors peak/valleys that are at the inferred_spot_size distance from them.
    """
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # Peaks
    diff = np.diff(peak_idxs)
    mask = np.logical_and(diff < (1 + size_threshold) * inferred_spot_size,
                          diff > (1 - size_threshold) * inferred_spot_size)
    mask1 = np.concatenate((mask, [False]))
    mask2 = np.concatenate(([False], mask))
    rank_peaks = np.add(mask1, mask2, dtype=np.int64)

    # Valleys
    diff = np.diff(valley_idxs)
    mask = np.logical_and(diff < (1 + size_threshold) * inferred_spot_size,
                          diff > (1 - size_threshold) * inferred_spot_size)
    mask1 = np.concatenate((mask, [False]))
    mask2 = np.concatenate(([False], mask))
    rank_valleys = np.add(mask1, mask2, dtype=np.int64)

    # Selecting best peak/valley from combined analysis
    idxs = np.concatenate((peak_idxs, valley_idxs))
    ranks = np.concatenate((rank_peaks, rank_valleys))
    sorted_idxs = np.argsort(idxs)
    idxs = idxs[sorted_idxs]
    ranks = ranks[sorted_idxs]
    ranks = smooth(ranks, 3)

    peak_ranks = ranks[np.isin(idxs, peak_idxs)]
    valley_ranks = ranks[np.isin(idxs, valley_idxs)]

    return peak_idxs[np.argmax(peak_ranks)], valley_idxs[np.argmax(valley_ranks)]


def refine_peaks(idxs, best, inferred_spot_size, image_height, threshold=0.1):
    """
    Refines the peaks/valleys using the best predicted peak/valley and the inferred spot size.
    """
    cleaned = np.array([best])
    pred = 1
    while pred > 0:
        ref = cleaned[0]
        pred = ref - inferred_spot_size
        mask = np.logical_and(idxs < pred * (1 + threshold), idxs > pred * (1 - threshold))
        if True in mask:
            pred = idxs[mask][0]
        cleaned = np.concatenate(([pred], cleaned))

    pred = 1
    while pred < image_height:
        ref = cleaned[-1]
        pred = ref + inferred_spot_size
        mask = np.logical_and(idxs < pred * (1 + threshold), idxs > pred * (1 - threshold))
        if True in mask:
            pred = idxs[mask][0]
        cleaned = np.concatenate((cleaned, [pred]))

    # Removing peaks/valleys outside of image
    cleaned = cleaned[cleaned >= 0]
    cleaned = cleaned[cleaned <= image_height]

    return cleaned


if __name__ == '__main__':
    plate_detector_save = "model_saves/Plate_detection.pt"
    phage_counter_save = "model_saves/Counter_phages.pt"
    # image_path = "data/lab_raw_good/20200204_115135.jpg"
    image_path = "data/lab_raw_good/20200204_115534.jpg"
    show_intermediate = False

    # --- Plate detection part ---
    image, detector_output = plate_detection(image_path, plate_detector_save)
    idxs = utils.batch_cleanup_boxes(
        detector_output["boxes"], detector_output["scores"], detector_output["labels"], 0.15)
    detector_output_cleaned = {k: v[idxs] for k, v in detector_output.items()}
    if show_intermediate:
        utils.plot_plate_detector(image, detector_output)

    # --- Extraction of images from box detection ---
    detector_images = plate_extraction(image, detector_output_cleaned)
    mean_column_width = np.mean([im.shape[2] for im in detector_images["phage_columns"]])

    # --- Extraction of spot from phage columns according to grayscale histogram
    for image in detector_images["phage_columns"]:
        # image = detector_images["phage_columns"][0]
        image = image.cpu().numpy().transpose((2, 1, 0))
        image = np.sum(image, axis=2)
        image_smoothed = scipy.ndimage.gaussian_filter(image, 25)
        nb_dilution_from_shape = image.shape[1] / image.shape[0]

        hist = np.sum(image_smoothed, axis=0)
        min_peak_distance = 0.7 * mean_column_width
        peak_idxs, valley_idxs = find_raw_peaks_and_valleys(hist, min_peak_distance)

        plt.figure()
        plt.plot(hist)
        plt.plot(peak_idxs, hist[peak_idxs], 'r.')
        plt.plot(valley_idxs, hist[valley_idxs], 'g.')

        # Selection of the size of plaques
        inferred_spot_size = estimate_spot_size(peak_idxs, valley_idxs, mean_column_width)

        # Selecting best peak and valley
        best_peak, best_valley = best_peak_and_valley(peak_idxs, valley_idxs, inferred_spot_size)
        plt.plot(best_peak, hist[best_peak], 'r.', markersize=10)
        plt.plot(best_valley, hist[best_valley], 'g.', markersize=10)

        # Cleaning all peaks and valleys
        cleaned_valleys = refine_peaks(valley_idxs, best_valley, inferred_spot_size, image.shape[1])
        cleaned_peaks = refine_peaks(peak_idxs, best_peak, inferred_spot_size, image.shape[1])

        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.plot(cleaned_valleys, np.zeros_like(cleaned_valleys), "g+")
        plt.plot(cleaned_peaks, np.zeros_like(cleaned_peaks), "r+")
