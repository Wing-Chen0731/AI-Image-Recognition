"""Check the runtime used by the Lesson 6/7 project.

Run from the repository root after activating the project conda environment:
    python tests/test_env.py
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"


def check_imports() -> bool:
    required = (
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("flask", "Flask"),
        ("ultralytics", "Ultralytics"),
    )
    all_ok = True
    print("Checking required packages...")
    for module_name, display_name in required:
        try:
            module = importlib.import_module(module_name)
            try:
                package_version = version(
                    "Pillow" if module_name == "PIL" else display_name
                )
            except PackageNotFoundError:
                package_version = getattr(module, "__version__", "installed")
            print(f"[OK] {display_name}: {package_version}")
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] {display_name}: {exc}")
    return all_ok


def check_assets() -> bool:
    classifier_data = PROJECT_ROOT / "data" / "oxford_pet_split" / "train"
    fallback_data = PROJECT_ROOT / "data" / "train"
    classifier_model = PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth"
    fallback_model = PROJECT_ROOT / "finetuned_mobilenet.pth"
    detector_model = PROJECT_ROOT / "yolov8n.pt"

    data_ready = classifier_data.is_dir() or fallback_data.is_dir()
    model_ready = classifier_model.is_file() or fallback_model.is_file()
    detector_ready = detector_model.is_file()

    print("Checking project assets...")
    print(f"[{'OK' if data_ready else 'FAIL'}] classifier training directory")
    print(f"[{'OK' if model_ready else 'FAIL'}] fine-tuned classifier checkpoint")
    print(
        f"[{'OK' if detector_ready else 'WARN'}] local YOLOv8 checkpoint "
        "(Ultralytics can download it on first detection)"
    )
    return data_ready and model_ready


def check_image() -> bool:
    try:
        import cv2

        candidates = list((PROJECT_ROOT / "data").rglob("*.jpg"))
        if not candidates:
            candidates = list((PROJECT_ROOT / "images").rglob("*.jpg"))
        if not candidates:
            print("[WARN] no JPG sample found for image-read check")
            return True

        image = cv2.imread(str(candidates[0]))
        if image is None:
            print(f"[FAIL] OpenCV could not read {candidates[0]}")
            return False
        height, width = image.shape[:2]
        print(f"[OK] image read: {width}x{height}")
        return True
    except Exception as exc:
        print(f"[FAIL] image-read check: {exc}")
        return False


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    results = (check_imports(), check_assets(), check_image())
    if all(results):
        print("Environment check passed.")
        return 0
    print("Environment check failed. Activate the correct conda environment and install requirements.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
