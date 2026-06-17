"""Flask web application for the image recognition project.

Run from the repository root after installing Flask:
    python app/web_app.py

The app loads the fine-tuned MobileNetV3 checkpoint once and reuses it for
every request. It serves a compact upload-and-predict UI that matches the
lesson 6 Web integration courseware.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    from flask import Flask, jsonify, render_template, request, url_for
    from werkzeug.utils import secure_filename
except ModuleNotFoundError:  # pragma: no cover - import path for documentation builds
    Flask = None  # type: ignore[assignment]
    jsonify = render_template = request = url_for = None  # type: ignore[assignment]
    secure_filename = None  # type: ignore[assignment]

try:
    from app.model_loader import FineTunedLoader, MobileNetV3Loader
except ModuleNotFoundError:  # Allows: python app/web_app.py
    from model_loader import FineTunedLoader, MobileNetV3Loader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "oxford_pet_split"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth"
STATIC_FOLDER = PROJECT_ROOT / "static"
TEMPLATE_FOLDER = PROJECT_ROOT / "templates"
UPLOAD_FOLDER = STATIC_FOLDER / "uploads"
IMAGE_SIZE = 224

app = (
    Flask(__name__, static_folder=str(STATIC_FOLDER), template_folder=str(TEMPLATE_FOLDER))
    if Flask is not None
    else None
)
if app is not None:
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

_model = None
_class_names: list[str] | None = None
_metadata: dict[str, str] = {}


def resolve_data_root() -> Path:
    """Return the first dataset root that contains a train split."""

    for root in (DEFAULT_DATA_ROOT, PROJECT_ROOT / "data"):
        train_dir = root / "train"
        if train_dir.exists():
            return root
    raise FileNotFoundError(
        "No dataset split found. Expected data/oxford_pet_split/train or data/train."
    )


@lru_cache(maxsize=1)
def load_class_names() -> tuple[str, ...]:
    """Load class names from the ImageFolder training split once."""

    train_dir = resolve_data_root() / "train"
    dataset = ImageFolder(train_dir)
    return tuple(dataset.classes)


def resolve_model_path() -> Path:
    """Pick a fine-tuned checkpoint if one is available."""

    for candidate in (DEFAULT_MODEL_PATH, PROJECT_ROOT / "finetuned_mobilenet.pth"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No fine-tuned checkpoint found. Expected models/oxford_pet_mobilenet_epoch1.pth "
        "or finetuned_mobilenet.pth."
    )


def get_model():
    """Load the fine-tuned model once and reuse it for all requests."""

    global _model, _metadata
    if _model is None:
        class_names = load_class_names()
        model_path = resolve_model_path()
        loader = FineTunedLoader(model_path, num_classes=len(class_names))
        _model = loader.load_model()
        _metadata = {
            "model_path": str(model_path),
            "data_root": str(resolve_data_root()),
            "class_count": str(len(class_names)),
        }
    return _model


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Convert an uploaded image into the tensor expected by MobileNetV3."""

    image = Image.open(image_path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return pipeline(image).unsqueeze(0)


def predict_image(image_path: Path, topk: int = 3) -> list[dict[str, float | str]]:
    """Run inference on a single image and return top-k class scores."""

    model = get_model()
    class_names = load_class_names()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits[0], dim=0)
        top_prob, top_idx = torch.topk(probs, k=min(topk, len(class_names)))

    results: list[dict[str, float | str]] = []
    for prob, idx in zip(top_prob, top_idx):
        results.append(
            {
                "label": class_names[int(idx)],
                "score": round(float(prob) * 100.0, 2),
            }
        )
    return results


def save_upload(upload) -> Path:
    """Persist an uploaded file with a sanitized filename."""

    if secure_filename is None:
        filename = Path(upload.filename).name
    else:
        filename = secure_filename(upload.filename)

    if not filename:
        raise ValueError("Uploaded file name is empty.")

    target = UPLOAD_FOLDER / filename
    upload.save(target)
    return target


if app is not None:

    @app.route("/")
    def index():
        class_names = load_class_names()
        model_path = resolve_model_path()
        return render_template(
            "index.html",
            model_path=str(model_path.relative_to(PROJECT_ROOT)),
            data_root=str(resolve_data_root().relative_to(PROJECT_ROOT)),
            class_count=len(class_names),
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空"}), 400

        try:
            image_path = save_upload(file)
            results = predict_image(image_path, topk=3)
            image_url = url_for("static", filename=f"uploads/{image_path.name}")
            return jsonify(
                {
                    "results": results,
                    "image_url": image_url,
                    "model_path": _metadata.get("model_path", ""),
                    "data_root": _metadata.get("data_root", ""),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flask image recognition app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if app is None:
        raise ModuleNotFoundError(
            "Flask is not installed. Install the requirements before running the web app."
        )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
