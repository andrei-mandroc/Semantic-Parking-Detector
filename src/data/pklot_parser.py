"""Parse PKLot parking-space annotations in original image coordinates."""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path


def _coordinate(value: str | None, space_id: str, axis: str) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Space {space_id} has an invalid {axis} coordinate") from None

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"Space {space_id} has an invalid {axis} coordinate")
    return int(number) if number.is_integer() else number


def _polygon_area(polygon: list[list[int | float]]) -> float:
    return abs(
        sum(
            point[0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * point[1]
            for index, point in enumerate(polygon)
        )
    ) / 2


def parse_annotation(xml_path: str | Path) -> tuple[list[dict], dict[str, bool]]:
    """Return parking-space polygons and available occupancy labels separately.

    Polygons use the original image's pixel coordinates. Missing or invalid
    occupancy labels are omitted from ground truth with a warning.
    """
    annotation_path = Path(xml_path)
    root = ET.parse(annotation_path).getroot()
    spaces: list[dict] = []
    ground_truth: dict[str, bool] = {}
    seen_ids: set[str] = set()

    for space in root.findall("space"):
        space_id = (space.get("id") or "").strip()
        if not space_id:
            raise ValueError(f"Parking space in {annotation_path} has no ID")
        if space_id in seen_ids:
            raise ValueError(f"Duplicate parking space ID {space_id} in {annotation_path}")
        seen_ids.add(space_id)

        contour = space.find("contour")
        if contour is None:
            raise ValueError(f"Space {space_id} has no contour in {annotation_path}")
        points = contour.findall("point")
        if len(points) < 3:
            raise ValueError(f"Space {space_id} needs at least 3 contour points")

        polygon = [
            [
                _coordinate(point.get("x"), space_id, "x"),
                _coordinate(point.get("y"), space_id, "y"),
            ]
            for point in points
        ]
        if _polygon_area(polygon) == 0:
            raise ValueError(f"Space {space_id} has a zero-area contour")
        spaces.append({"id": space_id, "polygon": polygon})

        occupied = space.get("occupied")
        if occupied in {"0", "1"}:
            ground_truth[space_id] = occupied == "1"
        else:
            warnings.warn(
                f"Space {space_id} has no valid occupied label in {annotation_path}",
                UserWarning,
                stacklevel=2,
            )

    return spaces, ground_truth
