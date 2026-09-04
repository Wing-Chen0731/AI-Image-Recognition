"""Object detection abstractions and YOLOv8 implementation.

CPSC 210 mapping:
- ObjectDetector is the interface used by the Web layer.
- YOLOv8Detector is one replaceable implementation of that interface.
- DetectionResult is a small value object for moving detection data around.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionResult:
    """One detected object in an image."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "label": self.label,
            "confidence": round(self.confidence * 100.0, 2),
        }


class ObjectDetector(ABC):
    """Interface for object detectors."""

    @abstractmethod
    def detect(
        self,
        image_path: str | Path,
        conf_threshold: float | None = None,
    ) -> list[DetectionResult]:
        """Detect objects in one image and return normalized result objects."""


class YOLOv8Detector(ObjectDetector):
    """YOLOv8 object detector backed by Ultralytics."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        conf_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0.")

        config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        # Force Ultralytics to keep its settings inside the project. This avoids
        # permission problems with a read-only user profile or another machine.
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        # Do not let a classroom request silently run pip or depend on network access.
        os.environ["YOLO_AUTOINSTALL"] = "false"

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is not installed. Run `pip install -r requirements.txt` "
                "inside the conda environment before using detection."
            ) from exc

        self.model_path = str(model_path)
        self.default_conf_threshold = conf_threshold
        self.model = YOLO(self.model_path)

    @staticmethod
    def _validate_threshold(conf_threshold: float) -> float:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0.")
        return conf_threshold

    def detect(
        self,
        image_path: str | Path,
        conf_threshold: float | None = None,
    ) -> list[DetectionResult]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        threshold = (
            self.default_conf_threshold
            if conf_threshold is None
            else self._validate_threshold(conf_threshold)
        )
        results = self.model(str(path), conf=threshold, verbose=False)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        detections: list[DetectionResult] = []
        for box in boxes:
            confidence = float(box.conf[0].item())
            if confidence < threshold:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            label = str(result.names[class_id])
            detections.append(
                DetectionResult(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    label=label,
                    confidence=confidence,
                )
            )

        return detections
