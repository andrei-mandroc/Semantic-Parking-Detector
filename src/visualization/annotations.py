"""Draw PKLot annotation geometry for visual inspection."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


AnnotationMode = Literal["geometry", "ground-truth"]

GEOMETRY_COLOR = (255, 255, 0)
FREE_COLOR = (0, 180, 0)
OCCUPIED_COLOR = (0, 0, 255)
UNKNOWN_COLOR = (160, 160, 160)


def render_annotations(
    image: np.ndarray,
    spaces: list[dict],
    ground_truth: dict[str, bool],
    mode: AnnotationMode = "geometry",
) -> np.ndarray:
    """Draw space IDs and contours without changing the input BGR image."""
    if mode not in {"geometry", "ground-truth"}:
        raise ValueError(f"Unknown annotation mode: {mode}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have three BGR channels")

    height, width = image.shape[:2]
    result = image.copy()
    title = "GEOMETRY ONLY" if mode == "geometry" else "GROUND TRUTH"
    cv2.putText(result, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(result, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for space in spaces:
        space_id = str(space["id"])
        polygon = np.asarray(space["polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"Space {space_id} has an invalid polygon")
        if not np.isfinite(polygon).all() or (polygon < 0).any():
            raise ValueError(f"Space {space_id} has invalid coordinates")
        if (polygon[:, 0] >= width).any() or (polygon[:, 1] >= height).any():
            raise ValueError(f"Space {space_id} is outside image bounds")

        points = polygon.astype(np.int32)
        color = GEOMETRY_COLOR
        if mode == "ground-truth":
            label = ground_truth.get(space_id)
            if label is None:
                color = UNKNOWN_COLOR
            elif label:
                color = OCCUPIED_COLOR
            else:
                color = FREE_COLOR

        cv2.polylines(result, [points], isClosed=True, color=(0, 0, 0), thickness=4)
        cv2.polylines(result, [points], isClosed=True, color=color, thickness=2)

        center_x, center_y = points.mean(axis=0).astype(int)
        text_size, _ = cv2.getTextSize(space_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_origin = (int(center_x - text_size[0] / 2), int(center_y + text_size[1] / 2))
        cv2.putText(result, space_id, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(result, space_id, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result
