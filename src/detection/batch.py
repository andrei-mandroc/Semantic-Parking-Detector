"""Batch inference pipeline for vehicle detection across multiple images."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Iterable

import cv2
import numpy as np

from src.detection.yolo_detector import VehicleDetector
from src.visualization.render import render_detections

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(source: str | Path) -> list[Path]:
    """Collect image paths from a directory, CSV manifest, or text file."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {source}")

    if path.is_dir():
        return sorted([
            p for p in path.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
        ])

    if path.suffix.lower() == ".csv":
        image_paths: list[Path] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            col_idx = 0
            if header:
                # Look for column name containing 'image' or 'path'
                for idx, col_name in enumerate(header):
                    if "image" in col_name.lower() or "path" in col_name.lower():
                        col_idx = idx
                        break
                else:
                    # If first row was data, check it
                    if Path(header[0]).suffix.lower() in SUPPORTED_EXTENSIONS:
                        image_paths.append(Path(header[0]))

            for row in reader:
                if row and len(row) > col_idx and row[col_idx].strip():
                    img_p = Path(row[col_idx].strip())
                    # Resolve relative to manifest directory if not absolute and not existing from cwd
                    if not img_p.is_absolute() and not img_p.exists():
                        candidate = path.parent / img_p
                        if candidate.exists():
                            img_p = candidate
                    image_paths.append(img_p)
        return image_paths

    # Plain text file (one path per line)
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return [Path(line) for line in lines]


def run_batch_detection(
    detector: VehicleDetector,
    image_paths: Iterable[Path | str],
    output_dir: Path | str | None = None,
    save_annotated: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run detection sequentially across images using a single detector instance.

    Returns:
        A tuple of:
          - records: list of dicts with per-image detection statistics.
          - summary: dict with aggregate statistics.
    """
    records: list[dict[str, Any]] = []
    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    total_images = 0
    successful_images = 0
    failed_images = 0
    total_vehicles = 0
    total_infer_time_ms = 0.0

    for item in image_paths:
        img_path = Path(item)
        total_images += 1

        record: dict[str, Any] = {
            "image_path": str(img_path),
            "model": detector.model_path,
            "confidence_threshold": detector.conf,
            "image_width": 0,
            "image_height": 0,
            "vehicle_count": 0,
            "inference_ms": 0.0,
            "status": "error",
            "error_message": "",
            "boxes": [],
        }

        if not img_path.is_file():
            record["error_message"] = "File not found"
            failed_images += 1
            records.append(record)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            record["error_message"] = "Cannot decode image"
            failed_images += 1
            records.append(record)
            continue

        h, w = image.shape[:2]
        record["image_width"] = w
        record["image_height"] = h

        try:
            start_time = time.perf_counter()
            detailed = detector.predict_detailed(image)
            infer_ms = (time.perf_counter() - start_time) * 1000

            boxes = [d["box"] for d in detailed]
            record["vehicle_count"] = len(boxes)
            record["inference_ms"] = round(infer_ms, 2)
            record["boxes"] = boxes
            record["detections"] = detailed
            record["status"] = "ok"

            successful_images += 1
            total_vehicles += len(boxes)
            total_infer_time_ms += infer_ms

            if save_annotated and out_path:
                annotated = render_detections(image, boxes)
                cv2.putText(
                    annotated,
                    f"Vehicles: {len(boxes)} | {infer_ms:.1f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                save_file = out_path / f"{img_path.stem}_detected{img_path.suffix}"
                cv2.imwrite(str(save_file), annotated)

        except Exception as exc:
            record["error_message"] = str(exc)
            failed_images += 1

        records.append(record)

    avg_infer_ms = round(total_infer_time_ms / successful_images, 2) if successful_images > 0 else 0.0

    summary = {
        "model": detector.model_path,
        "confidence_threshold": detector.conf,
        "total_images": total_images,
        "successful_images": successful_images,
        "failed_images": failed_images,
        "total_vehicles_detected": total_vehicles,
        "total_inference_time_ms": round(total_infer_time_ms, 2),
        "avg_inference_time_ms": avg_infer_ms,
    }

    return records, summary


def export_batch_results(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path | str,
    csv_filename: str = "summary.csv",
    save_json: bool = True,
) -> tuple[Path, Path | None]:
    """Export batch results to CSV and optional JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / csv_filename
    fieldnames = [
        "image_path",
        "model",
        "confidence_threshold",
        "image_width",
        "image_height",
        "vehicle_count",
        "inference_ms",
        "status",
        "error_message",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    json_path: Path | None = None
    if save_json:
        json_path = out_path / "detections.json"
        data = {
            "summary": summary,
            "records": records,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return csv_path, json_path
