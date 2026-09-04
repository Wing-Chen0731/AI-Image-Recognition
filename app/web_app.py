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
import importlib.util
import time
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
    from app.exceptions import ModelDownloadError
except ModuleNotFoundError:  # Allows: python app/web_app.py
    from model_loader import FineTunedLoader
    from object_detector import YOLOv8Detector
    from preprocess import draw_detections
    from exceptions import ModelDownloadError


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
MAX_UPLOAD_AGE_SECONDS = 24 * 60 * 60
MAX_UPLOAD_FILES = 100

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
_metadata: dict[str, str] = {}


def resolve_data_root() -> Path:
    """Return the first dataset root that contains a train split."""

    for root in (DEFAULT_DATA_ROOT, PROJECT_ROOT / "data"):
        train_dir = root / "train"
        if train_dir.is_dir():
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


def display_path(path: str | Path) -> str:
    """Return a project-relative path for UI and API responses."""

    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def classification_status() -> dict[str, str | int | bool]:
    """Describe whether the classification assets are ready without loading a model."""

    data_root = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.is_dir() else PROJECT_ROOT / "data"
    model_path = DEFAULT_MODEL_PATH
    try:
        data_root = resolve_data_root()
        class_count = len(load_class_names())
        model_path = resolve_model_path()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return {
            "ready": False,
            "data_root": display_path(data_root),
            "class_count": 0,
            "model_path": display_path(model_path),
            "message": str(exc),
        }

    return {
        "ready": True,
        "data_root": display_path(data_root),
        "class_count": class_count,
        "model_path": display_path(model_path),
        "message": "分类数据集与模型已就绪。",
    }


def detection_status() -> dict[str, str | bool]:
    """Describe the YOLO asset without downloading or loading it on page open."""

    detector_model = resolve_detector_model_path()
    ultralytics_installed = importlib.util.find_spec("ultralytics") is not None
    if not ultralytics_installed:
        ready = False
        message = "未安装 Ultralytics，请先执行 pip install -r requirements.txt。"
    elif Path(detector_model).is_file():
        ready = True
        message = "YOLOv8 权重已在项目中，可直接检测。"
    else:
        ready = True
        message = "首次使用检测时，Ultralytics 会尝试下载 yolov8n.pt。"
    return {
        "ready": ready,
        "model_path": display_path(detector_model),
        "message": message,
    }


def cleanup_uploads() -> None:
    """Keep classroom upload artifacts bounded and remove files older than one day."""

    def try_remove(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            # A file may still be held by an image viewer or antivirus scanner.
            # Skipping it keeps one locked temporary file from stopping Flask.
            if app is not None:
                app.logger.debug("Unable to remove temporary file %s: %s", path.name, exc)

    now = time.time()
    candidates = []
    for path in UPLOAD_FOLDER.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            try:
                modified_at = path.stat().st_mtime
            except OSError:
                continue
            if now - modified_at > MAX_UPLOAD_AGE_SECONDS:
                try_remove(path)
            else:
                candidates.append((modified_at, path))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, path in candidates[MAX_UPLOAD_FILES:]:
        try_remove(path)


cleanup_uploads()


def get_model():
    """Load the fine-tuned classifier once and reuse it for all requests."""

    global _model, _metadata
    if _model is None:
        class_names = load_class_names()
        model_path = resolve_model_path()
        loader = FineTunedLoader(model_path, num_classes=len(class_names))
        _model = loader.load_model()
        _metadata = {
            "model_path": display_path(model_path),
            "data_root": display_path(resolve_data_root()),
            "class_count": str(len(class_names)),
        }
    return _model


def get_detector():
    """Load the YOLOv8 detector once; confidence is selected per request."""

    global _detector
    if _detector is None:
        _detector = YOLOv8Detector(
            model_path=resolve_detector_model_path(),
            conf_threshold=DEFAULT_DETECTION_THRESHOLD,
        )
    return _detector


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Convert an uploaded image into the tensor expected by MobileNetV3."""

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
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

    detector = get_detector()
    detections = detector.detect(image_path, conf_threshold=conf_threshold)
    rendered_path = draw_detections(image_path, detections, output_dir=UPLOAD_FOLDER)
    cleanup_uploads()
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
    """Persist and validate an uploaded image with a sanitized unique filename."""

    cleanup_uploads()

    original_name = Path(upload.filename or "").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("只支持 JPG、PNG、BMP、WEBP 等常见图片格式。")

    # Keep the extension from the original name. Werkzeug removes non-ASCII
    # basename characters, so a Chinese-only filename could otherwise become
    # just "jpg" and fail the extension check.
    if secure_filename is None:
        safe_stem = "upload"
    else:
        safe_stem = Path(secure_filename(original_name)).stem or "upload"
        safe_stem = secure_filename(safe_stem) or "upload"

    if not safe_stem:
        raise ValueError("文件名为空。")

    target = UPLOAD_FOLDER / f"{uuid4().hex}_{safe_stem}{suffix}"
    upload.save(str(target))
    try:
        with Image.open(target) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError) as exc:
        target.unlink(missing_ok=True)
        raise ValueError("上传文件不是有效图片，或图片文件已损坏。") from exc
    return target


def image_url(image_path: Path) -> str:
    """Build a browser URL for an image under static/uploads."""

    return url_for("static", filename=f"uploads/{image_path.name}")


def inference_error_response(exc: Exception):
    """Convert expected inference failures into useful, safe API responses."""

    if isinstance(exc, ValueError):
        status_code = 400
        message = str(exc)
    elif isinstance(exc, ModelDownloadError):
        app.logger.exception("Model loading failed")
        status_code = 503
        message = "模型权重无法加载，请确认项目文件完整，并查看终端日志。"
    elif isinstance(exc, ModuleNotFoundError):
        status_code = 503
        message = str(exc)
    elif isinstance(exc, FileNotFoundError):
        status_code = 503
        message = "项目所需的数据或模型文件不存在，请先运行环境检查。"
    else:
        app.logger.exception("Inference request failed")
        status_code = 500
        message = "服务器处理图片时发生错误，请查看终端日志。"
    return jsonify({"error": message}), status_code


if app is not None:

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify({"error": "图片不能超过 16 MB。"}), 413

    @app.route("/")
    def index():
        classifier = classification_status()
        detector = detection_status()
        return render_template(
            "index.html",
            model_path=classifier["model_path"],
            detector_model=detector["model_path"],
            data_root=classifier["data_root"],
            class_count=classifier["class_count"],
            classification_ready=classifier["ready"],
            classification_message=classifier["message"],
            detection_ready=detector["ready"],
            detection_message=detector["message"],
            detection_threshold=DEFAULT_DETECTION_THRESHOLD,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件。"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空。"}), 400

        image_path = None
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
            if image_path is not None:
                image_path.unlink(missing_ok=True)
            return inference_error_response(exc)

    @app.route("/detect", methods=["POST"])
    def detect():
        if "file" not in request.files:
            return jsonify({"error": "没有上传文件。"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "文件名为空。"}), 400

        image_path = None
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
                    "detector_model": display_path(resolve_detector_model_path()),
                }
            )
        except Exception as exc:
            if image_path is not None:
                image_path.unlink(missing_ok=True)
            return inference_error_response(exc)


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
