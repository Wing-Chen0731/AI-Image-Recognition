"""Complete MobileNetV3 fine-tuning script for Lesson 4.

Run from the repository root:
    python app/finetune.py --data-dir data --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    from app.exceptions import ClassCountMismatchError, DataDirectoryNotFoundError
    from app.model_loader import build_finetune_model
    from app.transforms import train_transform, val_transform
except ModuleNotFoundError:  # Allows: python app/finetune.py
    from exceptions import ClassCountMismatchError, DataDirectoryNotFoundError
    from model_loader import build_finetune_model
    from transforms import train_transform, val_transform


def progress(iterable: Iterable, label: str) -> Iterable:
    """Use tqdm when it is installed, otherwise return the original iterable."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=label, leave=False)


def validate_data_dir(data_dir: Path) -> None:
    """Check ImageFolder preconditions before training starts."""

    if not data_dir.exists():
        raise DataDirectoryNotFoundError(f"Data directory does not exist: {data_dir}")

    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.exists():
            raise DataDirectoryNotFoundError(
                f"Expected dataset split is missing: {split_dir}"
            )
        class_dirs = [path for path in split_dir.iterdir() if path.is_dir()]
        if not class_dirs:
            raise DataDirectoryNotFoundError(
                f"Dataset split has no class folders: {split_dir}"
            )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch and return loss and accuracy."""

    model.train()
    total_loss = 0.0
    correct = 0

    for images, labels in progress(dataloader, "train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()

    size = len(dataloader.dataset)
    return total_loss / size, correct / size


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate for one epoch and return loss and accuracy."""

    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in progress(dataloader, "val"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

    size = len(dataloader.dataset)
    return total_loss / size, correct / size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV3.")
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Expected class count. Defaults to the number found in data/train.",
    )
    parser.add_argument(
        "--output",
        default="finetuned_mobilenet.pth",
        help="Where to save the model state_dict.",
    )
    parser.add_argument(
        "--unfreeze-features",
        action="store_true",
        help="Train the feature extractor as well as the classifier head.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    validate_data_dir(data_dir)

    train_set = ImageFolder(data_dir / "train", train_transform)
    val_set = ImageFolder(data_dir / "val", val_transform)
    discovered_classes = len(train_set.classes)

    if args.num_classes is not None and discovered_classes != args.num_classes:
        raise ClassCountMismatchError(
            f"Expected {args.num_classes} classes, found {discovered_classes}: "
            f"{train_set.classes}"
        )

    num_classes = args.num_classes or discovered_classes
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_finetune_model(
        num_classes=num_classes,
        freeze_features=not args.unfreeze_features,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
    )

    print(f"Device: {device}")
    print(f"Classes: {train_set.class_to_idx}")
    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")

    best_val_acc = 0.0
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
