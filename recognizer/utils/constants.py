from enum import Enum


class Constants:
    IMAGE_NET_MEANS = [0.485, 0.456, 0.406]
    IMAGE_NET_STDS = [0.229, 0.224, 0.225]


class DatasetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class SupportedLRSchedulers(Enum):
    EXPONENTIAL = "ExponentialLR"
    REDUCE_PLATEAU = "ReduceLROnPlateau"
