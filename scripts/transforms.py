import utils
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import torch


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
    def __init__(self, mode=0, proba=0.5, max_size=1333):
        self.mode = mode
        self.proba = proba
        self.max_size = max_size

    def __call__(self, image, target):
        if self.mode == 0:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 1:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    A.HorizontalFlip(p=self.proba),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 2:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    A.HorizontalFlip(p=self.proba),
                    A.RandomBrightnessContrast(p=self.proba),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 3:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    A.HorizontalFlip(p=self.proba),
                    A.ColorJitter(p=self.proba),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 4:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    A.HorizontalFlip(p=self.proba),
                    A.RandomBrightnessContrast(p=self.proba),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=self.proba),
                             A.GaussNoise(0.1, p=self.proba),
                             A.MultiplicativeNoise([0.8, 1.2], elementwise=True, p=self.proba)]),
                    ToTensorV2()
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        if self.mode == 5:
            transform = A.Compose(
                [
                    A.LongestMaxSize(self.max_size),
                    A.HorizontalFlip(p=self.proba),
                    A.ColorJitter(p=self.proba),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 21), p=self.proba),
                             A.GaussNoise(0.1, p=self.proba),
                             A.MultiplicativeNoise([0.8, 1.2], elementwise=True, p=self.proba)]),
                    ToTensorV2()
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
        breakpoint()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # image_path = "data/plates_labeled/images/20200204_115031.jpg"
    # label_path = "data/plates_labeled/labels/20200204_115031.txt"
    # augmentation = PlateAlbumentation(0, proba=0.5)
    # check_dataset_augmentation(image_path, label_path, augmentation)

    image_path = "data/plates_labeled/test/images/20211112_103710.jpg"
    augment = A.LongestMaxSize(1333)
    image = utils.load_image_from_file(image_path, dtype="int")
    image = augment.__call__(image=image)["image"]
    from PIL import Image
    im = Image.fromarray(image)
