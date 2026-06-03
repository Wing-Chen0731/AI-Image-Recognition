"""Model loading helpers for inference and fine-tuning.

CPSC 210 mapping:
- ModelLoader is an abstraction for loading pretrained models.
- build_finetune_model extends MobileNetV3 for a custom class count while
  keeping the feature extractor closed to modification when it is frozen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

try:
    from app.exceptions import ModelDownloadError
except ModuleNotFoundError:  # Allows: python app/classifier.py
    from exceptions import ModelDownloadError


class ModelLoader(ABC):
    """Interface for loading a ready-to-use model."""

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load and return a torch.nn.Module."""


class MobileNetV3Loader(ModelLoader):
    """Load MobileNetV3-Large with ImageNet weights for inference."""

    def load_model(self) -> nn.Module:
        try:
            from torchvision.models import (
                MobileNet_V3_Large_Weights,
                mobilenet_v3_large,
            )

            model = mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
        except Exception as exc:
            raise ModelDownloadError(
                f"Failed to load MobileNetV3-Large weights: {exc}"
            ) from exc

        model.eval()
        return model


class ResNet18Loader(ModelLoader):
    """Load ResNet-18 with ImageNet weights for inference."""

    def load_model(self) -> nn.Module:
        try:
            from torchvision.models import ResNet18_Weights, resnet18

            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise ModelDownloadError(f"Failed to load ResNet-18 weights: {exc}") from exc

        model.eval()
        return model


def build_finetune_model(
    num_classes: int,
    freeze_features: bool = True,
) -> nn.Module:
    """Build a MobileNetV3 model for custom image classification.

    Args:
        num_classes: Number of output classes in the custom dataset.
        freeze_features: If True, only the classifier head is trained.
    """

    if num_classes < 2:
        raise ValueError("Fine-tuning requires at least two classes.")

    try:
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    except Exception as exc:
        raise ModelDownloadError(
            f"Failed to load MobileNetV3-Large for fine-tuning: {exc}"
        ) from exc

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
