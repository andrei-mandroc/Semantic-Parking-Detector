"""Geometry-based parking occupancy rules using vehicle bounding boxes."""
from __future__ import annotations

import cv2
import numpy as np


def _prepare_spaces(image_shape: tuple[int, int], spaces: list[dict]) -> list[tuple[str, np.ndarray]]:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")

    prepared: list[tuple[str, np.ndarray]] = []
    seen_ids: set[str] = set()
    for space in spaces:
        space_id = str(space["id"])
        if not space_id or space_id in seen_ids:
            raise ValueError(f"Space ID must be nonempty and unique: {space_id}")
        seen_ids.add(space_id)

        polygon = np.asarray(space["polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"Space {space_id} needs at least 3 coordinate pairs")
        if not np.isfinite(polygon).all():
            raise ValueError(f"Space {space_id} has non-finite coordinates")
        if (
            (polygon < 0).any()
            or (polygon[:, 0] >= width).any()
            or (polygon[:, 1] >= height).any()
        ):
            raise ValueError(f"Space {space_id} is outside image bounds")

        points = polygon.astype(np.int32)
        if cv2.contourArea(points) == 0:
            raise ValueError(f"Space {space_id} has a zero-area polygon")
        prepared.append((space_id, points))

    return prepared


def _clipped_boxes(image_shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    height, width = image_shape
    clipped: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    return clipped


def coverage_scores(image_shape: tuple[int, int],spaces: list[dict],boxes: list[tuple[int, int, int, int]],) -> dict[str, float]:
    """Return the fraction of each polygon covered by the union of vehicle boxes."""
    prepared = _prepare_spaces(image_shape, spaces)
    vehicle_mask = np.zeros(image_shape, dtype=np.uint8)
    for x1, y1, x2, y2 in _clipped_boxes(image_shape, boxes):
        vehicle_mask[y1:y2, x1:x2] = 255

    scores: dict[str, float] = {}
    for space_id, polygon in prepared:
        space_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.fillPoly(space_mask, [polygon], 255)
        area = cv2.countNonZero(space_mask)
        covered = cv2.countNonZero(cv2.bitwise_and(space_mask, vehicle_mask))
        scores[space_id] = covered / area
    return scores


def classify_spaces(
    image_shape: tuple[int, int],
    spaces: list[dict],
    boxes: list[tuple[int, int, int, int]],
    threshold: float = 0.15,
) -> dict[str, bool]:
    """Classify by polygon coverage, preserving the original baseline rule."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    scores = coverage_scores(image_shape, spaces, boxes)
    return {space_id: score >= threshold for space_id, score in scores.items()}


def classify_spaces_by_center(
    image_shape: tuple[int, int],
    spaces: list[dict],
    boxes: list[tuple[int, int, int, int]],
) -> dict[str, bool]:
    """Classify a space as occupied when a clipped vehicle-box center is inside it."""
    prepared = _prepare_spaces(image_shape, spaces)
    clipped_boxes = _clipped_boxes(image_shape, boxes)
    return {
        space_id: any(
            cv2.pointPolygonTest(
                polygon,
                ((x1 + x2) / 2, (y1 + y2) / 2),
                False,
            ) >= 0
            for x1, y1, x2, y2 in clipped_boxes
        )
        for space_id, polygon in prepared
    }
