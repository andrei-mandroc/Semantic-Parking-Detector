"""CLI test script for vehicle instance segmentation (Task AI-05)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.segmentation_detector import (
    VehicleSegmentationDetector,
    render_segmentation_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO instance segmentation on a single image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image file.")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="YOLO segmentation model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold in (0.0, 1.0].")
    parser.add_argument("--output", type=Path, default=None, help="Path to save annotated segmentation output.")
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

    print(f"Loading segmentation detector with model '{args.model}' (conf={args.conf})...")
    start_init = time.perf_counter()
    detector = VehicleSegmentationDetector(model_path=args.model, conf=args.conf)
    init_ms = (time.perf_counter() - start_init) * 1000

    h, w = image.shape[:2]
    print(f"Running instance segmentation on {args.image} ({w}x{h} px)...")
    start_infer = time.perf_counter()
    detections = detector.predict_detailed(image)
    infer_ms = (time.perf_counter() - start_infer) * 1000

    print(f"Segmentation completed in {infer_ms:.1f} ms (model load: {init_ms:.1f} ms)")
    print(f"Detected vehicles: {len(detections)}")

    for i, d in enumerate(detections, start=1):
        box = d["box"]
        poly = d.get("polygon")
        n_pts = len(poly) if poly is not None else 0
        score = d["confidence"]
        print(f"  Vehicle #{i}: box={box}, mask points={n_pts}, conf={score:.3f}")

    annotated = render_segmentation_overlay(image, detections)
    cv2.putText(
        annotated,
        f"Segmented Vehicles: {len(detections)} | Time: {infer_ms:.1f}ms",
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
        output_path = output_dir / f"{args.image.stem}_segmentation{args.image.suffix}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated segmentation image saved to: {output_path}")


if __name__ == "__main__":
    main()
