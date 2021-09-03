import random
import torch
import torchvision
import numpy as np
import albumentations as A

from dataset import H5Dataset, LabH5Dataset
from torchvision.transforms import functional as F
from albumentations.pytorch import ToTensorV2
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


class CounterToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), F.to_tensor(target)


class CounterAlbumentation(object):
    def __init__(self, train=True, mode=0):
        self.train = train
        self.mode = mode

    def __call__(self, image, target):
        if self.train:
            if self.mode == 0:
                transform = A.Compose(
                    [
                        A.Flip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2()
                    ],
                    additional_targets={"target": "image"})
            if self.mode == 1:
                transform = A.Compose(
                    [
                        A.Flip(p=0.5),
                        A.Rotate(p=0.5),
                        ToTensorV2()
                    ],
                    additional_targets={"target": "image"})
            if self.mode == 2:
                transform = A.Compose(
                    [A.Compose(
                        [
                            A.ColorJitter(),
                            A.GaussianBlur(blur_limit=(3, 5)),
                            A.GaussNoise(0.002)
                        ]),
                     A.Compose(
                        [
                            A.Flip(p=0.5),
                            A.Rotate(p=0.5),
                            # A.RandomResizedCrop(256, 256, scale=(0.5, 0.9), ratio=(1, 1), p=0.5),
                            ToTensorV2()
                        ],
                        additional_targets={"target": "image"})
                     ]
                )
        else:
            transform = A.Compose([ToTensorV2()],
                                  additional_targets={"target": "image"})

        transformed = transform(image=image, target=target)
        return transformed["image"], transformed["target"]


class PlateAlbumentation(object):
    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, image, target):
        if self.mode == 0:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 1:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 2:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10),
                    A.GaussianBlur(blur_limit=(3, 21), p=0.5),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 3:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10),
                    A.GaussianBlur(blur_limit=(3, 21), p=0.5),
                    A.ColorJitter(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        transformed = transform(image=image, bboxes=target["boxes"], class_labels=target["labels"])
        return transformed["image"], {"boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32),
                                      "labels": torch.tensor(transformed["class_labels"], dtype=torch.int64)}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Phage_colonies_folder = "data/phage_plates/"
    # dataset = LabH5Dataset(Phage_colonies_folder + "train.h5", None)
    #
    # # image, boxes, labels = dataset.images[0], dataset.boxes[0], dataset.labels[0]
    # image, boxes, labels = dataset.__getitem__(0)

    # true = label.sum()
    #
    # for ii in range(10):
    #     transform = CounterAlbumentation()
    #     transim, translab = transform(image, label)
    #     transim = np.transpose(transim, (2, 1, 0))
    #     translab = np.transpose(translab, (2, 1, 0))
    #
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(transim)
    #     axs[1].imshow(translab)
    #     axs[1].set_xlabel(f"{translab.sum()} {true}")
    # plt.show()

    dataset = LabH5Dataset("data/phage_plates/train.h5", PlateAlbumentation(mode=3))
    for ii in range(5):
        image, target = dataset.__getitem__(0)
        utils.plot_image_target(image, target)
    plt.show()
