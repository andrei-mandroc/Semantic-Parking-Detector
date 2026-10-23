"""Simple geometry-based occupancy baseline; tune and validate on PKLot."""
from __future__ import annotations

import cv2
import numpy as np


def classify_spaces(
    image_shape: tuple[int, int],
    spaces: list[dict],
    boxes: list[tuple[int, int, int, int]],
    threshold: float = 0.15,
) -> dict[str, bool]:
    """A space is occupied if detected-vehicle boxes cover enough of its polygon.

    Notes: overlapping boxes are treated as a union; this is a baseline and
    should be evaluated against PKLot ground truth, not assumed to be accurate.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    height, width = image_shape
    result: dict[str, bool] = {}
    for space in spaces:
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.asarray(space["polygon"], dtype=np.int32)
        if polygon.shape[0] < 3:
            raise ValueError(f"Space {space['id']} needs at least 3 points")
        cv2.fillPoly(mask, [polygon], 255)
        area = cv2.countNonZero(mask)
        vehicle_mask = np.zeros_like(mask)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                vehicle_mask[y1:y2, x1:x2] = 255
        covered = cv2.countNonZero(cv2.bitwise_and(mask, vehicle_mask))
        result[str(space["id"])] = area > 0 and covered / area >= threshold
    return result
