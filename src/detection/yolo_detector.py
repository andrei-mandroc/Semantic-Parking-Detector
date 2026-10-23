"""YOLO-based vehicle detection (COCO vehicle classes)."""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

# COCO IDs: car, motorcycle, bus, truck.
VEHICLE_CLASSES = [2, 3, 5, 7]


class VehicleDetector:
    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        result = self.model.predict(source=image, classes=VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
        if result.boxes is None:
            return []
        return [tuple(map(int, box)) for box in result.boxes.xyxy.cpu().numpy()]
