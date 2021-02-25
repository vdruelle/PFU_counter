import random
import torch
import torchvision

from torchvision.transforms import functional as F


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
