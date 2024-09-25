import logging
import random

from torchvision import datasets, transforms
from typing import Tuple
from torch.utils.data import DataLoader, random_split, dataset, Dataset, Subset
from .utils.constants import Constants, DatasetType

logger = logging.getLogger(__name__)


class ToFloat16Transform:
    def __call__(self, sample):
        """
        Call the function on the input sample and return half of it.
        """
        return sample.half()


class ProjectDataset:
    @staticmethod
    def transform(
        resize_shape: Tuple[int, int], dataset_type: DatasetType, f32: bool = False
    ) -> transforms.Compose:
        """
        Transformations to be applied to the dataset
        :param f32:
        :param dataset_type: DatasetType
        :param resize_shape: Tuple[int, int]
        :return: transforms.Compose
        """
        processors = []
        if dataset_type == DatasetType.TRAIN:
            processors.append(
                transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0))
            )
            processors.append(
                transforms.RandomAffine(
                    degrees=(-20, 20), translate=(0.1, 0.1), scale=None
                )
            )
            processors.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            )

        if dataset_type == DatasetType.VALIDATION or dataset_type == DatasetType.TEST:
            processors.append(transforms.Resize(resize_shape))

        quantization_options = []

        if f32:
            quantization_options.append(ToFloat16Transform())

        return transforms.Compose(
            [
                *processors,
                transforms.ToTensor(),
                *quantization_options,
                # Normalization parameters based on ImageNet dataset's mean and standard deviation which RESNET34 was
                # trained on
                #
                # INFO: If you are training a model from scratch, you should calculate the mean and standard deviation
                #       of the RGB channels from the dataset in question and use those values for normalization
                transforms.Normalize(
                    mean=Constants.IMAGE_NET_MEANS, std=Constants.IMAGE_NET_STDS,
                ),
            ]
        )

    @staticmethod
    def _get(
        training_dataset_path: str,
        testing_dataset_path: str,
        validation_split: float,
        resize_shape: Tuple[int, int],
        f32: bool = False,
    ) -> Tuple[dataset.Subset, dataset.Subset, datasets.ImageFolder]:
        """
        Get train, validation and test datasets
        :param training_dataset_path:
        :param testing_dataset_path:
        :param validation_split:
        :param resize_shape:
        :return: Tuple[dataset.Subset, dataset.Subset, datasets.ImageFolder]
        """
        full_dataset = datasets.ImageFolder(
            root=training_dataset_path,
            transform=ProjectDataset.transform(resize_shape, DatasetType.TRAIN, f32),
        )

        train_dataset, validation_dataset = random_split(
            full_dataset,
            [
                int(len(full_dataset) * (1 - validation_split)),
                int(len(full_dataset) * validation_split),
            ],
        )

        test_dataset = datasets.ImageFolder(
            root=testing_dataset_path,
            transform=ProjectDataset.transform(
                resize_shape=resize_shape, dataset_type=DatasetType.TEST, f32=f32
            ),
        )

        return train_dataset, validation_dataset, test_dataset

    @staticmethod
    def get_loaders(
        training_dataset_path: str = "../data/train",
        testing_dataset_path: str = "../data/test",
        batch_size: int = 64,
        num_workers: int = 4,
        validation_split: float = 0.2,
        resize_shape: Tuple[int, int] = (224, 224),
        f32: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation and test dataloaders
        :param f32:
        :param training_dataset_path:
        :param testing_dataset_path:
        :param batch_size:
        :param num_workers:
        :param validation_split:
        :param resize_shape:
        :return: Tuple[DataLoader, DataLoader, DataLoader]
        """
        train_dataset, val_dataset, test_dataset = ProjectDataset._get(
            training_dataset_path=training_dataset_path,
            testing_dataset_path=testing_dataset_path,
            validation_split=validation_split,
            resize_shape=resize_shape,
            f32=f32,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader
