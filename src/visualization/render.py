"""Draw detections and parking-space occupancy on a copy of an image."""
from __future__ import annotations

import cv2
import numpy as np


def render_detections(
    image: np.ndarray,
    vehicles: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (255, 180, 0),
    thickness: int = 2,
) -> np.ndarray:
    result = image.copy()
    for i, (x1, y1, x2, y2) in enumerate(vehicles, start=1):
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        label = f"#{i}"
        cv2.putText(
            result,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
    return result


def render_parking_result(
    image: np.ndarray,
    spaces: list[dict],
    statuses: dict[str, bool],
    vehicles: list[tuple[int, int, int, int]],
) -> np.ndarray:
    result = image.copy()
    for x1, y1, x2, y2 in vehicles:
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 180, 0), 2)
    for space in spaces:
        polygon = np.asarray(space["polygon"], dtype=np.int32)
        occupied = statuses[str(space["id"])]
        color = (0, 0, 255) if occupied else (0, 180, 0)  # BGR
        cv2.polylines(result, [polygon], isClosed=True, color=color, thickness=2)
        x, y = polygon[0]
        label = f"{space['id']}: {'OCCUPIED' if occupied else 'FREE'}"
        cv2.putText(result, label, (int(x), max(15, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return result
