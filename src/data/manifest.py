"""Build reproducible PKLot manifests from local image and XML pairs."""

from __future__ import annotations

import random
import re
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2

from src.data.pklot_parser import parse_annotation


MANIFEST_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CAMERAS = {"PUCPR", "UFPR04", "UFPR05"}
WEATHERS = {"sunny", "cloudy", "rainy"}
DATE_PATTERN = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
MANIFEST_FIELDS = (
    "manifest_version", "image_path", "xml_path", "file_name", "camera", "weather",
    "date", "session", "split",
)


def _metadata(path: Path, dataset_root: Path, camera: str | None, weather: str | None) -> tuple[str, str, str, str]:
    relative_path = path.relative_to(dataset_root)
    parts = (dataset_root.name, *relative_path.parts)
    tokens = [token for part in parts for token in re.split(r"[^A-Za-z0-9]+", part)]
    inferred_camera = next((token.upper() for token in tokens if token.upper() in CAMERAS), None)
    inferred_weather = next((token.lower() for token in tokens if token.lower() in WEATHERS), None)
    date_match = next((match for part in reversed(parts) if (match := DATE_PATTERN.search(part))), None)
    date = date_match.group() if date_match else ""
    session = date or f"directory:{relative_path.parent.as_posix()}"
    return (
        inferred_camera or (camera.upper() if camera else "unknown"),
        inferred_weather or (weather.lower() if weather else "unknown"),
        date,
        session,
    )


def assign_splits(rows: list[dict[str, str]], seed: int = 42) -> list[dict[str, str]]:
    """Assign whole capture days or sessions to train, validation, and test."""
    sessions = sorted({row["session"] for row in rows})
    random.Random(seed).shuffle(sessions)
    count = len(sessions)
    if count >= 3:
        train_count = min(count - 2, max(1, round(count * 0.70)))
        validation_count = min(count - train_count - 1, max(1, round(count * 0.15)))
    else:
        train_count = min(count, 1)
        validation_count = count - train_count
    split_by_session = {
        session: ("train" if index < train_count else
                  "validation" if index < train_count + validation_count else "test")
        for index, session in enumerate(sessions)
    }
    return [{**row, "split": split_by_session[row["session"]]} for row in rows]


def build_manifest(
    dataset_root: str | Path,
    seed: int = 42,
    camera: str | None = None,
    weather: str | None = None,
) -> tuple[list[dict[str, str]], dict]:
    """Index valid pairs using paths relative to dataset_root and report rejected files."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    images: dict[str, list[Path]] = {}
    annotations: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(key, []).append(path)
        elif path.suffix.lower() == ".xml":
            annotations.setdefault(key, []).append(path)

    image_only = sorted(key for key in images if key not in annotations)
    xml_only = sorted(key for key in annotations if key not in images)
    rows: list[dict[str, str]] = []
    invalid_pairs: list[dict[str, str]] = []
    missing_labels = 0
    for key in sorted(images.keys() & annotations.keys()):
        image_paths = images[key]
        xml_paths = annotations[key]
        xml_path = xml_paths[0]
        relative_xml = xml_path.relative_to(root).as_posix()
        relative_image = image_paths[0].relative_to(root).as_posix()
        try:
            if len(image_paths) != 1:
                raise ValueError("Multiple images share the same annotation")
            if len(xml_paths) != 1:
                raise ValueError("Multiple annotations share the same image")
            image = cv2.imread(str(image_paths[0]))
            if image is None:
                raise ValueError("Image cannot be decoded")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                spaces, ground_truth = parse_annotation(xml_path)
            if not spaces:
                raise ValueError("Annotation contains no parking spaces")
            height, width = image.shape[:2]
            if any(
                not (0 <= x < width and 0 <= y < height)
                for space in spaces for x, y in space["polygon"]
            ):
                raise ValueError("Parking polygon lies outside image bounds")
            missing_labels += len(spaces) - len(ground_truth)
        except (OSError, ValueError, ET.ParseError, cv2.error) as error:
            invalid_pairs.append({"image_path": relative_image, "xml_path": relative_xml, "reason": str(error)})
            continue

        inferred_camera, inferred_weather, date, session = _metadata(image_paths[0], root, camera, weather)
        rows.append({
            "manifest_version": str(MANIFEST_VERSION),
            "image_path": relative_image,
            "xml_path": relative_xml,
            "file_name": image_paths[0].name,
            "camera": inferred_camera,
            "weather": inferred_weather,
            "date": date,
            "session": session,
        })

    rows = assign_splits(rows, seed)
    report = {
        "manifest_version": MANIFEST_VERSION,
        "seed": seed,
        "valid_pairs": len(rows),
        "images_without_xml": [path.relative_to(root).as_posix() for key in image_only for path in images[key]],
        "xml_without_images": [path.relative_to(root).as_posix() for key in xml_only for path in annotations[key]],
        "invalid_pairs": invalid_pairs,
        "spaces_without_valid_labels": missing_labels,
        "by_camera": dict(sorted(Counter(row["camera"] for row in rows).items())),
        "by_weather": dict(sorted(Counter(row["weather"] for row in rows).items())),
        "by_date": dict(sorted(Counter(row["date"] or "unknown" for row in rows).items())),
        "by_split": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
    }
    return rows, report
