"""Unit tests for local fine-tuned checkpoint loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from app.model_loader import FineTunedLoader, build_finetune_model


class MinimalMobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(1, 1))
        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(4, 1000),
        )


class FineTunedLoaderTests(unittest.TestCase):
    def test_non_pretrained_build_passes_weights_none(self) -> None:
        source_model = MinimalMobileNet()
        with patch(
            "torchvision.models.mobilenet_v3_large",
            return_value=source_model,
        ) as model_factory:
            model = build_finetune_model(
                num_classes=3,
                freeze_features=True,
                pretrained=False,
            )

        model_factory.assert_called_once_with(weights=None)
        self.assertEqual(model.classifier[3].out_features, 3)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))

    def test_local_checkpoint_does_not_request_pretrained_weights(self) -> None:
        source_model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pth"
            torch.save(source_model.state_dict(), checkpoint)

            target_model = nn.Linear(2, 2)
            with patch(
                "app.model_loader.build_finetune_model",
                return_value=target_model,
            ) as build_model:
                loaded_model = FineTunedLoader(
                    checkpoint,
                    num_classes=2,
                    freeze_features=True,
                ).load_model()

        build_model.assert_called_once_with(
            num_classes=2,
            freeze_features=True,
            pretrained=False,
        )
        self.assertIs(loaded_model, target_model)
        self.assertFalse(loaded_model.training)


if __name__ == "__main__":
    unittest.main()
