"""Smoke test script for vehicle detection on a single image (Task AI-01)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import VehicleDetector
from src.visualization.render import render_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a single image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or weights name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold in (0.0, 1.0].")
    parser.add_argument("--output", type=Path, default=None, help="Path to save annotated output image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Error: failed to load image from {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading detector with model '{args.model}' (conf={args.conf})...")
    start_init = time.perf_counter()
    detector = VehicleDetector(model_path=args.model, conf=args.conf)
    init_time_ms = (time.perf_counter() - start_init) * 1000

    height, width = image.shape[:2]
    print(f"Running detection on {args.image} ({width}x{height} px)...")
    start_infer = time.perf_counter()
    vehicles = detector.predict(image)
    infer_time_ms = (time.perf_counter() - start_infer) * 1000

    print(f"Detection completed in {infer_time_ms:.1f} ms (model load: {init_time_ms:.1f} ms)")
    print(f"Detected vehicles: {len(vehicles)}")

    for i, (x1, y1, x2, y2) in enumerate(vehicles, start=1):
        box_w = x2 - x1
        box_h = y2 - y1
        print(f"  Vehicle #{i}: [{x1}, {y1}, {x2}, {y2}] ({box_w}x{box_h} px)")

    annotated = render_detections(image, vehicles)
    summary_text = f"Vehicles: {len(vehicles)} | Time: {infer_time_ms:.1f}ms"
    cv2.putText(
        annotated,
        summary_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    if args.output:
        output_path = args.output
    else:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.image.stem}_detection{args.image.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
