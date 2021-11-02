import os
import sys
import pathlib
import numpy as np
import torch
import torchvision
import utils
import math
import scipy
import matplotlib.pyplot as plt

from dataset import PlateDataset
from transforms import PlateAlbumentation
from torch.utils.tensorboard import SummaryWriter
from model import PlateDetector


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


if __name__ == '__main__':
    image_path = "data/phage_spots/test/images/20200204_115031_3_0.png"
    label_path = "data/phage_spots/test/labels/20200204_115031_3_0_labels.png"

    label = utils.load_image_from_file(label_path)
    image = utils.load_image_from_file(image_path)
    density = gaussian_filter_density(label)

    plt.figure()
    plt.imshow(density)
    plt.figure()
    plt.imshow(image)
