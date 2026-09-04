"""Evaluate a fine-tuned MobileNetV3 checkpoint on an ImageFolder split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_loader import FineTunedLoader  # noqa: E402
from app.transforms import val_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pet classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "oxford_pet_split",
        help="Dataset root containing the validation split.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth",
        help="Fine-tuned checkpoint to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    val_dir = args.data_dir / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")

    dataset = ImageFolder(val_dir, transform=val_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = resolve_device(args.device)
    model = FineTunedLoader(args.model, num_classes=len(dataset.classes)).load_model()
    model = model.to(device)

    correct = 0
    total = 0
    class_correct = [0] * len(dataset.classes)
    class_total = [0] * len(dataset.classes)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            matches = predictions.eq(labels)

            correct += int(matches.sum().item())
            total += int(labels.numel())
            for label, matched in zip(labels.tolist(), matches.tolist()):
                class_total[label] += 1
                class_correct[label] += int(matched)

    accuracy = correct / total if total else 0.0
    per_class = sorted(
        (
            class_correct[index] / class_total[index] if class_total[index] else 0.0,
            class_name,
            class_correct[index],
            class_total[index],
        )
        for index, class_name in enumerate(dataset.classes)
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.model}")
    print(f"Validation samples: {total}")
    print(f"Top-1 accuracy: {accuracy:.2%} ({correct}/{total})")
    print("Lowest per-class accuracies:")
    for score, class_name, class_hits, class_size in per_class[:5]:
        print(f"  {class_name}: {score:.2%} ({class_hits}/{class_size})")


if __name__ == "__main__":
    main()
