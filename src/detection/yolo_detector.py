"""YOLO-based vehicle detection with configurable parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

# COCO vehicle class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck.
# These correspond to the typical motorized road vehicles present in parking lots.
DEFAULT_VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_CLASSES = DEFAULT_VEHICLE_CLASSES


class VehicleDetector:
    """Configurable wrapper around Ultralytics YOLO for vehicle detection."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        conf: float = 0.25,
        imgsz: int | tuple[int, int] | None = None,
        classes: list[int] | None = None,
        device: str | int | None = None,
    ) -> None:
        self.conf = self._validate_conf(conf)
        self.imgsz = self._validate_imgsz(imgsz)
        self.classes = self._validate_classes(classes)
        self.device = device
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)

    @staticmethod
    def _validate_conf(conf: float) -> float:
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 < conf <= 1.0):
            raise ValueError(f"Confidence threshold must be a number in (0.0, 1.0], got {conf!r}")
        return float(conf)

    @staticmethod
    def _validate_imgsz(imgsz: int | tuple[int, int] | None) -> int | tuple[int, int] | None:
        if imgsz is None:
            return None
        if isinstance(imgsz, int) and not isinstance(imgsz, bool):
            if imgsz <= 0:
                raise ValueError(f"Image size must be a positive integer, got {imgsz}")
            return imgsz
        if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in imgsz):
                raise ValueError(f"Image size tuple must contain two positive integers, got {imgsz}")
            return (imgsz[0], imgsz[1])
        raise ValueError(f"Image size must be an int or a pair of positive ints, got {imgsz!r}")

    @staticmethod
    def _validate_classes(classes: list[int] | None) -> list[int]:
        if classes is None:
            return list(DEFAULT_VEHICLE_CLASSES)
        if not isinstance(classes, (list, tuple)):
            raise TypeError(f"Classes must be a list or tuple of integers, got {type(classes).__name__}")
        if not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in classes):
            raise ValueError(f"Classes must contain non-negative integers, got {classes!r}")
        return list(classes)

    def _run_inference(self, image: np.ndarray, conf: float | None = None) -> Any:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be a numpy.ndarray, got {type(image).__name__}")
        if image.size == 0 or image.ndim not in (2, 3):
            raise ValueError(f"Invalid image array with shape {getattr(image, 'shape', None)}")

        active_conf = self._validate_conf(conf) if conf is not None else self.conf

        predict_kwargs: dict[str, Any] = {
            "source": image,
            "classes": self.classes,
            "conf": active_conf,
            "verbose": False,
        }
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        return results[0] if results else None

    def predict(
        self,
        image: np.ndarray,
        conf: float | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Detect vehicles and return bounding boxes as (x1, y1, x2, y2) tuples."""
        result = self._run_inference(image, conf=conf)
        if result is None or result.boxes is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        if len(boxes_xyxy) == 0:
            return []
        return [tuple(map(int, box)) for box in boxes_xyxy]

    def predict_detailed(
        self,
        image: np.ndarray,
        conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Detect vehicles with bounding boxes, confidence scores, and class IDs."""
        result = self._run_inference(image, conf=conf)
        if result is None or result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return []

        confs = (
            result.boxes.conf.cpu().numpy()
            if result.boxes.conf is not None
            else np.zeros(len(boxes), dtype=float)
        )
        classes = (
            result.boxes.cls.cpu().numpy()
            if result.boxes.cls is not None
            else np.zeros(len(boxes), dtype=int)
        )

        detections: list[dict[str, Any]] = []
        for box, score, cls_id in zip(boxes, confs, classes):
            detections.append({
                "box": tuple(map(int, box)),
                "confidence": float(score),
                "class_id": int(cls_id),
            })
        return detections
