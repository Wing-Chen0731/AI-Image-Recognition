"""Fine-tune MobileNetV3 on a small ImageFolder dataset.

Expected dataset layout:

data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/

Run from the repository root:
    python app/train.py --data-dir data --epochs 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is unavailable
    tqdm = None

try:
    from app.dataset import create_dataloaders
    from app.model_loader import build_finetune_model
except ModuleNotFoundError:  # Allows: python app/train.py
    from dataset import create_dataloaders
    from model_loader import build_finetune_model


def iter_progress(iterable: Iterable, description: str) -> Iterable:
    """Wrap an iterable with tqdm when it is installed."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, leave=False)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in iter_progress(dataloader, f"Epoch {epoch} train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model and return average loss and accuracy."""

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in iter_progress(dataloader, "Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV3 on an ImageFolder dataset."
    )
    parser.add_argument("--data-dir", default="data", help="Dataset root directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count. Use 0 on Windows for classroom demos.",
    )
    parser.add_argument(
        "--output",
        default="finetuned_mobilenet.pth",
        help="Path for the saved model state_dict.",
    )
    parser.add_argument(
        "--unfreeze-features",
        action="store_true",
        help="Update the feature extractor as well as the classifier head.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_to_idx)

    model = build_finetune_model(
        num_classes=num_classes,
        freeze_features=not args.unfreeze_features,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
    )

    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    best_val_acc = 0.0
    output_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path}")


if __name__ == "__main__":
    main()
