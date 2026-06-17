from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
LABEL_PATTERN = re.compile(r"^(?P<label>.+)_\d+$")


def parse_label(path: Path) -> str:
    match = LABEL_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"Cannot parse class label from filename: {path.name}")
    return match.group("label")


def copy_split(files: list[Path], split_root: Path, label: str) -> None:
    class_dir = split_root / label
    class_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, class_dir / src.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified train/val split for Oxford-IIIT Pet images."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/oxford_pet_split"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before writing the new split.",
    )
    args = parser.parse_args()

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(args.output_dir)

    by_label: dict[str, list[Path]] = defaultdict(list)
    for path in args.images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            by_label[parse_label(path)].append(path)

    if not by_label:
        raise ValueError(f"No image files found in: {args.images_dir}")

    rng = random.Random(args.seed)
    train_root = args.output_dir / "train"
    val_root = args.output_dir / "val"
    summary: list[tuple[str, int, int]] = []

    for label in sorted(by_label):
        files = sorted(by_label[label])
        rng.shuffle(files)
        val_count = max(1, round(len(files) * args.val_ratio))
        val_files = files[:val_count]
        train_files = files[val_count:]

        copy_split(train_files, train_root, label)
        copy_split(val_files, val_root, label)
        summary.append((label, len(train_files), len(val_files)))

    total_train = sum(item[1] for item in summary)
    total_val = sum(item[2] for item in summary)
    print(f"Created split at: {args.output_dir}")
    print(f"Classes: {len(summary)}")
    print(f"Train images: {total_train}")
    print(f"Val images: {total_val}")
    print()
    print("Per-class counts:")
    for label, train_count, val_count in summary:
        print(f"{label}: train={train_count}, val={val_count}")


if __name__ == "__main__":
    main()
