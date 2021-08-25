import random
import torch
import torchvision
import numpy as np
import albumentations as A

from dataset import H5Dataset
from torchvision.transforms import functional as F
import utils


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class GaussianBlur(object):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        # Using a tuple for sigma range means it selects randomly values in this range
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, target):
        image = torchvision.transforms.GaussianBlur(self.kernel_size, self.sigma).forward(image)
        return image, target


#### For the colony counter ####

class CounterRandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.fliplr(image)
            target = np.fliplr(target)
        return image.copy(), target.copy()  # copy necessary otherwise pytorch crash


class CounterRandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = np.flipud(image)
            target = np.flipud(target)
        return image.copy(), target.copy()  # copy necessary otherwise pytorch crash


class CounterNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(torch.from_numpy(image), mean=self.mean, std=self.std)
        # return image.copy(), target.copy()  # copy necessary otherwise pytorch crash
        return image, torch.from_numpy(target)  # copy necessary otherwise pytorch crash


class CounterAlbumentation(object):
    def __call__(self, image, target):
        transform = A.Compose(
            [A.Flip(p=0.5),
             A.Rotate(p=0.5),
             A.RandomCrop(200, 200, p=0.5)
             ],
            additional_targets={"target": "image"}
        )
        transformed = transform(image=image, target=target)
        return transformed["image"], transformed["target"]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Phage_colonies_folder = "data/phage_colonies/"
    dataset = H5Dataset(Phage_colonies_folder + "train.h5", None)

    image, label = dataset.images[0], dataset.labels[0]
    image = np.transpose(image, (2, 1, 0))
    label = np.transpose(label, (2, 1, 0))

    for ii in range(10):
        transform = CounterAlbumentation()
        transim, translab = transform(image, label)

        fig, axs = plt.subplots(1,2)
        axs[0].imshow(transim)
        axs[1].imshow(translab)
    plt.show()
