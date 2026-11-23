"""CLI script for batch vehicle detection on images or manifests (Task AI-03)."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.batch import collect_image_paths, export_batch_results, run_batch_detection
from src.detection.yolo_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch vehicle detection on image folders or manifests.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to image directory, CSV manifest, or text file listing image paths.",
    )
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or weights name.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold in (0.0, 1.0].")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size (e.g. 640).")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. 'cpu', '0'.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save CSV/JSON reports and optional annotated images.",
    )
    parser.add_argument("--save-images", action="store_true", help="Save annotated images with bounding boxes.")
    parser.add_argument("--no-json", action="store_true", help="Do not export detections.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        image_paths = collect_image_paths(args.source)
    except Exception as exc:
        print(f"Error collecting images: {exc}", file=sys.stderr)
        sys.exit(1)

    if not image_paths:
        print(f"No valid images found in source: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / f"batch_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} images to process.")
    print(f"Loading YOLO model '{args.model}' (conf={args.conf})...")
    detector = VehicleDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("Running batch inference...")
    records, summary = run_batch_detection(
        detector=detector,
        image_paths=image_paths,
        output_dir=output_dir,
        save_annotated=args.save_images,
    )

    csv_path, json_path = export_batch_results(
        records=records,
        summary=summary,
        output_dir=output_dir,
        save_json=not args.no_json,
    )

    print("\n" + "=" * 55)
    print("BATCH DETECTION SUMMARY")
    print("=" * 55)
    print(f"  Model:                   {summary['model']}")
    print(f"  Confidence threshold:    {summary['confidence_threshold']}")
    print(f"  Total images processed:  {summary['total_images']}")
    print(f"  Successful:              {summary['successful_images']}")
    print(f"  Failed / unreadable:     {summary['failed_images']}")
    print(f"  Total vehicles detected: {summary['total_vehicles_detected']}")
    print(f"  Total inference time:    {summary['total_inference_time_ms']} ms")
    print(f"  Avg inference per image: {summary['avg_inference_time_ms']} ms")
    print("-" * 55)
    print(f"  CSV summary saved to:    {csv_path}")
    if json_path:
        print(f"  JSON details saved to:   {json_path}")
    if args.save_images:
        print(f"  Annotated images saved:  {output_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
