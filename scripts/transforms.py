import random
import torch
import torchvision
import numpy as np
import albumentations as A

from dataset import H5Dataset, LabH5Dataset, PlateDataset
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
    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, image, target):
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
                [A.Compose([A.RandomBrightnessContrast()]),
                 A.Compose([
                     A.Flip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     ToTensorV2()],
                     additional_targets={"target": "image"})
                 ],
            )
        if self.mode == 2:
            transform = A.Compose(
                [A.Compose([A.ColorJitter()]),
                 A.Compose([
                     A.Flip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     ToTensorV2()],
                     additional_targets={"target": "image"})
                 ],
            )
        if self.mode == 3:
            transform = A.Compose(
                [A.Compose([
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                             A.GaussNoise(0.002, p=0.5),
                             A.MultiplicativeNoise([0.95, 1.05], elementwise=True, p=0.5)]),
                    A.ColorJitter()]),
                 A.Compose([
                     A.Flip(p=0.5),
                     A.RandomRotate90(p=0.5),
                     ToTensorV2()],
                     additional_targets={"target": "image"})
                 ],
            )

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
                    A.Rotate(limit=2),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 2:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=2),
                    A.RandomBrightnessContrast(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 3:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=2),
                    A.RandomBrightnessContrast(),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=0.5),
                             A.GaussNoise(0.1, p=0.5)]),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 4:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=0.5),
                             A.GaussNoise(0.1, p=0.5),
                             A.MultiplicativeNoise([0.8, 1.2], elementwise=True, p=1)]),
                    A.HueSaturationValue(20, 0.2, 0.1),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 5:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50ms
                    A.RandomBrightnessContrast(),  # 183ms
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=0.5),  # 105ms
                             A.GaussNoise(0.1, p=0.5),  # 1266ms
                             A.MultiplicativeNoise([0.8, 1.2], elementwise=True, p=0.5)]),  # 460ms
                    A.HueSaturationValue(20, 0.2, 0.1),  # 494ms
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1)),  # 186ms
                    ToTensorV2()  # 34ms
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 6:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50ms
                    A.Rotate(limit=2),
                    A.RandomBrightnessContrast(),  # 183ms
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=0.5),  # 105ms
                             A.GaussNoise(0.1, p=0.5),  # 1266ms
                             A.MultiplicativeNoise([0.8, 1.2], elementwise=True, p=0.5)]),  # 460ms
                    A.HueSaturationValue(20, 0.2, 0.1),  # 494ms
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1)),  # 186ms
                    ToTensorV2()  # 34ms
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 7:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50ms
                    A.Rotate(limit=2),
                    A.RandomBrightnessContrast(),  # 183ms
                    A.GaussianBlur(blur_limit=(3, 21), p=0.5),  # 105ms
                    A.HueSaturationValue(20, 0.2, 0.1),  # 494ms
                    A.RandomShadow(shadow_roi=(0, 0, 1, 1)),  # 186ms
                    ToTensorV2()  # 34ms
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        transformed = transform(image=image, bboxes=target["boxes"], class_labels=target["labels"])
        return transformed["image"], {"boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32),
                                      "labels": torch.tensor(transformed["class_labels"], dtype=torch.int64)}


class CounterBoxAlbumentation(object):
    def __init__(self, mode=0):
        self.mode = mode

    def __call__(self, image, target):
        if self.mode == 0:
            transform = A.Compose(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 1:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 2:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 3:
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 11), p=0.5),
                             A.GaussNoise(0.005, p=0.5),
                             A.MultiplicativeNoise([0.9, 1.1], elementwise=True, p=0.5)]),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        transformed = transform(image=image, bboxes=target["boxes"], class_labels=target["labels"])
        return transformed["image"], {"boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32),
                                      "labels": torch.tensor(transformed["class_labels"], dtype=torch.int64)}


def check_dataset_augmentation(image_path, label_path, augmentation, nb=10):
    """
    Plots images and augmented images.
    """
    image = utils.load_image_from_file(image_path)
    boxes, labels = utils.boxes_and_labels_from_file(label_path, image.shape[0], image.shape[1])
    target = {"boxes": boxes, "labels": labels}
    for ii in range(nb):
        augmented_image, augmented_target = augmentation.__call__(image, target)
        utils.plot_spot_boxes(augmented_image, augmented_target)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_path = "data/phage_spots_minimal/box_labeling/images/20200204_115042_3.jpg"
    label_path = "data/phage_spots_minimal/box_labeling/labels/20200204_115042_3.txt"
    augmentation = CounterBoxAlbumentation(3)
    check_dataset_augmentation(image_path, label_path, augmentation)
