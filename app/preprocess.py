"""Image post-processing helpers for detection results."""

from __future__ import annotations

from pathlib import Path

try:
    from app.object_detector import DetectionResult
except ModuleNotFoundError:  # Allows direct script-style imports from app/
    from object_detector import DetectionResult


def draw_detections(
    image_path: str | Path,
    detections: list[DetectionResult],
    output_dir: str | Path | None = None,
) -> Path:
    """Draw detection boxes on the original image and save a rendered copy."""

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "opencv-python is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    source = Path(image_path)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Unable to read image file: {source}")

    height, width = image.shape[:2]
    for detection in detections:
        x1 = max(0, min(width - 1, int(detection.x1)))
        y1 = max(0, min(height - 1, int(detection.y1)))
        x2 = max(0, min(width - 1, int(detection.x2)))
        y2 = max(0, min(height - 1, int(detection.y2)))

        color = (39, 174, 96)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"{detection.label}: {detection.confidence * 100:.1f}%"
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_y1 = max(0, y1 - text_height - 8)
        label_y2 = max(text_height + 4, y1)
        cv2.rectangle(
            image,
            (x1, label_y1),
            (min(width - 1, x1 + text_width + 8), label_y2),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, max(text_height + 1, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    target_dir = Path(output_dir) if output_dir is not None else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}_detected{source.suffix or '.jpg'}"
    cv2.imwrite(str(target), image)
    return target
