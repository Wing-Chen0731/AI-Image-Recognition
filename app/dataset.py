"""Dataset and preprocessing utilities for image fine-tuning.

This module supports Lesson 4: dataset preparation and transfer learning.
It keeps preprocessing strategies separate from DataLoader construction so
training and validation transforms can vary without changing the dataset code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PreprocessingStrategy(ABC):
    """Strategy interface for image preprocessing transforms."""

    @abstractmethod
    def build(self) -> transforms.Compose:
        """Return a torchvision transform pipeline."""


class TrainingPreprocessingStrategy(PreprocessingStrategy):
    """Data augmentation used while training."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def build(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


class ValidationPreprocessingStrategy(PreprocessingStrategy):
    """Deterministic preprocessing used while validating."""

    def __init__(self, image_size: int = 224, resize_size: int = 256) -> None:
        self.image_size = image_size
        self.resize_size = resize_size

    def build(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


def validate_split_dir(split_dir: Path) -> None:
    """Validate an ImageFolder split such as data/train or data/val."""

    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"Dataset split is not a directory: {split_dir}")

    class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"Dataset split must contain class subdirectories: {split_dir}"
        )


def build_image_folder(
    split_dir: str | Path, strategy: PreprocessingStrategy
) -> datasets.ImageFolder:
    """Create an ImageFolder dataset with the supplied preprocessing strategy."""

    split_path = Path(split_dir)
    validate_split_dir(split_path)
    return datasets.ImageFolder(root=str(split_path), transform=strategy.build())


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create train/validation loaders for a data/train and data/val layout."""

    root = Path(data_dir)
    train_dataset = build_image_folder(
        root / "train", TrainingPreprocessingStrategy(image_size=image_size)
    )
    val_dataset = build_image_folder(
        root / "val", ValidationPreprocessingStrategy(image_size=image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.class_to_idx


def count_classes(data_dir: str | Path = "data") -> int:
    """Return the number of classes in data/train."""

    train_dir = Path(data_dir) / "train"
    validate_split_dir(train_dir)
    return len([path for path in train_dir.iterdir() if path.is_dir()])
