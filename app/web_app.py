"""Flask web application for the image recognition project.

Run from the repository root after installing dependencies:
    python app/web_app.py

Lesson 7 extends the lesson 6 classification app with an object-detection
workflow. The Web app now exposes two endpoints:
- /predict for image classification.
- /detect for YOLOv8 object detection.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

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
    from app.model_loader import FineTunedLoader
    from app.object_detector import YOLOv8Detector
    from app.preprocess import draw_detections
except ModuleNotFoundError:  # Allows: python app/web_app.py
    from model_loader import FineTunedLoader
    from object_detector import YOLOv8Detector
    from preprocess import draw_detections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "oxford_pet_split"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth"
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "yolov8n.pt"
STATIC_FOLDER = PROJECT_ROOT / "static"
TEMPLATE_FOLDER = PROJECT_ROOT / "templates"
UPLOAD_FOLDER = STATIC_FOLDER / "uploads"
IMAGE_SIZE = 224
DEFAULT_DETECTION_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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
_detector = None
_detector_threshold: float | None = None
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
    """Pick a fine-tuned classification checkpoint if one is available."""

    for candidate in (DEFAULT_MODEL_PATH, PROJECT_ROOT / "finetuned_mobilenet.pth"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No fine-tuned checkpoint found. Expected models/oxford_pet_mobilenet_epoch1.pth "
        "or finetuned_mobilenet.pth."
    )


def resolve_detector_model_path() -> str:
    """Use a local YOLOv8 weight file when present, otherwise let Ultralytics fetch it."""

    if DEFAULT_DETECTOR_MODEL.exists():
        return str(DEFAULT_DETECTOR_MODEL)
    return "yolov8n.pt"


def get_model():
    """Load the fine-tuned classifier once and reuse it for all requests."""

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


def get_detector(conf_threshold: float = DEFAULT_DETECTION_THRESHOLD):
    """Load the YOLOv8 detector once and reuse it while the threshold is unchanged."""

    global _detector, _detector_threshold
    if _detector is None or _detector_threshold != conf_threshold:
        _detector = YOLOv8Detector(
            model_path=resolve_detector_model_path(),
            conf_threshold=conf_threshold,
        )
        _detector_threshold = conf_threshold
    return _detector


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
    """Run classification on a single image and return top-k class scores."""

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


def detect_image(
    image_path: Path,
    conf_threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> tuple[list[dict[str, float | str]], Path]:
    """Run object detection and save a rendered image with boxes."""

    detector = get_detector(conf_threshold)
    detections = detector.detect(image_path)
    rendered_path = draw_detections(image_path, detections, output_dir=UPLOAD_FOLDER)
    return [detection.to_dict() for detection in detections], rendered_path


def parse_conf_threshold() -> float:
    """Parse and validate detection confidence threshold from form data."""

    raw_value = request.form.get("conf_threshold", str(DEFAULT_DETECTION_THRESHOLD))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度阈值必须是 0 到 1 之间的数字。") from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError("置信度阈值必须在 0 到 1 之间。")
    return value


def save_upload(upload) -> Path:
    """Persist an uploaded image with a sanitized unique filename."""

    original_name = Path(upload.filename or "").name
    if secure_filename is None:
        filename = original_name
    else:
        filename = secure_filename(original_name)

    if not filename:
        raise ValueError("文件名为空。")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("只支持 JPG、PNG、BMP、WEBP 等常见图片格式。")

    target = UPLOAD_FOLDER / f"{uuid4().hex}_{filename}"
    upload.save(target)
    return target


def image_url(image_path: Path) -> str:
    """Build a browser URL for an image under static/uploads."""

    return url_for("static", filename=f"uploads/{image_path.name}")


if app is not None:

    @app.route("/")
    def index():
        class_names = load_class_names()
        model_path = resolve_model_path()
        return render_template(
            "index.html",
            model_path=str(model_path.relative_to(PROJECT_ROOT)),
            detector_model=resolve_detector_model_path(),
            data_root=str(resolve_data_root().relative_to(PROJECT_ROOT)),
            class_count=len(class_names),
            detection_threshold=DEFAULT_DETECTION_THRESHOLD,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件。"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空。"}), 400

        try:
            image_path = save_upload(file)
            results = predict_image(image_path, topk=3)
            return jsonify(
                {
                    "results": results,
                    "image_url": image_url(image_path),
                    "model_path": _metadata.get("model_path", ""),
                    "data_root": _metadata.get("data_root", ""),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/detect", methods=["POST"])
    def detect():
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件。"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空。"}), 400

        try:
            conf_threshold = parse_conf_threshold()
            image_path = save_upload(file)
            detections, rendered_path = detect_image(image_path, conf_threshold)
            return jsonify(
                {
                    "detections": detections,
                    "count": len(detections),
                    "image_url": image_url(rendered_path),
                    "source_image_url": image_url(image_path),
                    "conf_threshold": conf_threshold,
                    "detector_model": resolve_detector_model_path(),
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
