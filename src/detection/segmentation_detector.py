"""YOLO-based instance segmentation for vehicle contour and mask detection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.detection.yolo_detector import DEFAULT_VEHICLE_CLASSES


class VehicleSegmentationDetector:
    """Configurable wrapper around Ultralytics YOLO segmentation models."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n-seg.pt",
        conf: float = 0.25,
        imgsz: int | tuple[int, int] | None = None,
        classes: list[int] | None = None,
        device: str | int | None = None,
    ) -> None:
        self.conf = self._validate_conf(conf)
        self.imgsz = imgsz
        self.classes = list(classes) if classes is not None else list(DEFAULT_VEHICLE_CLASSES)
        self.device = device
        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)

    @staticmethod
    def _validate_conf(conf: float) -> float:
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 < conf <= 1.0):
            raise ValueError(f"Confidence threshold must be a number in (0.0, 1.0], got {conf!r}")
        return float(conf)

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
        """Return vehicle bounding boxes for compatibility with the base detector interface."""
        result = self._run_inference(image, conf=conf)
        if result is None or result.boxes is None:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        if len(boxes_xyxy) == 0:
            return []
        return [tuple(map(int, box)) for box in boxes_xyxy]

    def predict_masks(
        self,
        image: np.ndarray,
        conf: float | None = None,
    ) -> list[np.ndarray]:
        """Return list of polygon coordinate arrays for detected vehicle contours."""
        result = self._run_inference(image, conf=conf)
        if result is None or result.masks is None or result.masks.xy is None:
            return []

        polygons: list[np.ndarray] = []
        for poly in result.masks.xy:
            if len(poly) >= 3:
                polygons.append(np.asarray(poly, dtype=np.int32))
        return polygons

    def predict_detailed(
        self,
        image: np.ndarray,
        conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return vehicle bounding boxes, polygon contours, confidence, and class IDs."""
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

        has_masks = result.masks is not None and result.masks.xy is not None
        mask_polys = result.masks.xy if has_masks else []

        detections: list[dict[str, Any]] = []
        for i, (box, score, cls_id) in enumerate(zip(boxes, confs, classes)):
            poly = None
            if i < len(mask_polys) and len(mask_polys[i]) >= 3:
                poly = np.asarray(mask_polys[i], dtype=np.int32)

            detections.append({
                "box": tuple(map(int, box)),
                "polygon": poly,
                "confidence": float(score),
                "class_id": int(cls_id),
            })
        return detections


def render_segmentation_overlay(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    alpha: float = 0.45,
    contour_color: tuple[int, int, int] = (0, 255, 255),
    fill_color: tuple[int, int, int] = (255, 140, 0),
) -> np.ndarray:
    """Render semi-transparent filled masks and polygon contours over the image."""
    result = image.copy()
    overlay = image.copy()

    for i, item in enumerate(detections, start=1):
        poly = item.get("polygon")
        if poly is not None and len(poly) >= 3:
            # Draw semi-transparent fill
            cv2.fillPoly(overlay, [poly], fill_color)
            # Draw solid contour boundary
            cv2.polylines(result, [poly], isClosed=True, color=contour_color, thickness=2)

            # Label on top point of polygon
            x_min, y_min = np.min(poly, axis=0)
            label = f"Veh #{i} ({item.get('confidence', 0):.2f})"
            cv2.putText(
                result,
                label,
                (int(x_min), max(15, int(y_min) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                contour_color,
                1,
                cv2.LINE_AA,
            )
        elif item.get("box"):
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(result, (x1, y1), (x2, y2), contour_color, 2)

    cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0, result)
    return result
