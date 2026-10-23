"""Entry point: detection + optional spot occupancy for a parking image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.detection.yolo_detector import VehicleDetector
from src.parking.occupancy import classify_spaces
from src.visualization.render import render_parking_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Parking Detector starter")
    parser.add_argument("--image", type=Path, required=True, help="Path to a parking-lot image")
    parser.add_argument("--config", type=Path, default=Path("configs/parking_spaces.json"))
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics pretrained weights")
    parser.add_argument("--threshold", type=float, default=0.15, help="Minimum spot area covered by detections")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.config.is_file():
        parser.error(f"Config not found: {args.config}")

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Image cannot be read: {args.image}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    spaces = config.get("spaces", [])
    if spaces and (image.shape[1], image.shape[0]) != (config["image_width"], config["image_height"]):
        parser.error("Image dimensions differ from slot configuration; adjust the polygons first")

    vehicles = VehicleDetector(args.model).predict(image)
    statuses = classify_spaces(image.shape[:2], spaces, vehicles, threshold=args.threshold)
    annotated = render_parking_result(image, spaces, statuses, vehicles)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.image.stem}_annotated.jpg"
    cv2.imwrite(str(output), annotated)

    print(f"Vehicles detected: {len(vehicles)}")
    if spaces:
        occupied = sum(statuses.values())
        print(f"Parking spaces: {len(spaces)} | occupied: {occupied} | free: {len(spaces)-occupied}")
    else:
        print("No parking-space polygons configured yet; vehicle detection only.")
    print(f"Saved result: {output.resolve()}")


if __name__ == "__main__":
    main()
