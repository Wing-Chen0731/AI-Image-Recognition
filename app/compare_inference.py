"""Compare pretrained vs fine-tuned MobileNetV3 on a single image.

Usage from repository root:
    conda run -n pytorch_env python app/compare_inference.py

Default image:
    images/cat.jpg

This script prints:
- ImageNet top-5 predictions from the pretrained model
- Custom class probabilities from the fine-tuned model
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from app.model_loader import FineTunedLoader, MobileNetV3Loader
except ModuleNotFoundError:  # Allows: python app/compare_inference.py
    from model_loader import FineTunedLoader, MobileNetV3Loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "images" / "cat.jpg"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "oxford_pet_split"


def load_image(image_path: Path) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return preprocess(image).unsqueeze(0)


def predict_pretrained(image_tensor: torch.Tensor) -> List[Tuple[str, float]]:
    loader = MobileNetV3Loader()
    model = loader.load_model()

    weights = None
    try:
        from torchvision.models import MobileNet_V3_Large_Weights

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    except Exception:
        pass

    categories: Sequence[str]
    if weights is not None:
        categories = weights.meta["categories"]
    else:
        categories = [str(i) for i in range(1000)]

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)

    return [(categories[idx], float(prob) * 100.0) for prob, idx in zip(top5_prob, top5_idx)]


def predict_finetuned(image_tensor: torch.Tensor, model_path: Path, data_dir: Path) -> List[Tuple[str, float]]:
    try:
        from torchvision.datasets import ImageFolder

        class_names = ImageFolder(data_dir / "train").classes
    except Exception as exc:
        raise RuntimeError(f"Unable to read class names from {data_dir / 'train'}: {exc}") from exc

    loader = FineTunedLoader(model_path, num_classes=len(class_names))
    model = loader.load_model()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits[0], dim=0)

    return [(class_names[i], float(probs[i]) * 100.0) for i in range(len(class_names))]


def print_block(title: str, rows: Sequence[Tuple[str, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for idx, (label, score) in enumerate(rows, start=1):
        print(f"{idx}. {label:30s} {score:6.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ImageNet and fine-tuned MobileNetV3 predictions."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image
    model_path = args.model
    data_dir = args.data_dir

    if not image_path.exists():
        raise FileNotFoundError(f"Default image missing: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned model missing: {model_path}. Run training first."
        )

    image_tensor = load_image(image_path)

    pretrained_results = predict_pretrained(image_tensor)
    finetuned_results = predict_finetuned(image_tensor, model_path, data_dir)

    print(f"Image: {image_path}")
    print(f"Fine-tuned model: {model_path}")
    print_block("Pretrained MobileNetV3 (ImageNet)", pretrained_results)
    print_block("Fine-tuned MobileNetV3 (custom classes)", finetuned_results)

    best_custom = max(finetuned_results, key=lambda item: item[1])
    print(f"\nFine-tuned top prediction: {best_custom[0]} ({best_custom[1]:.2f}%)")


if __name__ == "__main__":
    main()
