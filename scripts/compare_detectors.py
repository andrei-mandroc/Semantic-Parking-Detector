"""Script to benchmark and compare multiple YOLO detection configurations (Task AI-04)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import ultralytics

from src.detection.batch import collect_image_paths, run_batch_detection
from src.detection.yolo_detector import VehicleDetector

DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "yolo11n_conf0.15",
        "model": "yolo11n.pt",
        "conf": 0.15,
        "description": "High sensitivity (catches distant/partially visible cars)",
    },
    {
        "name": "yolo11n_conf0.25",
        "model": "yolo11n.pt",
        "conf": 0.25,
        "description": "Balanced baseline (default YOLO threshold)",
    },
    {
        "name": "yolo11n_conf0.40",
        "model": "yolo11n.pt",
        "conf": 0.40,
        "description": "Conservative (minimizes false positive vehicle detections)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare detection configurations on the same image set.")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "samples",
        help="Path to image directory, manifest, or list file.",
    )
    parser.add_argument(
        "--configs-json",
        type=Path,
        default=None,
        help="Path to custom JSON file specifying detector configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save comparison reports and visualizations.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated images for each configuration for visual comparison.",
    )
    return parser.parse_args()


def load_configurations(json_path: Path | None) -> list[dict[str, Any]]:
    if json_path is not None:
        if not json_path.is_file():
            raise FileNotFoundError(f"Configurations file not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Configurations JSON must contain a list of configuration objects.")
            return data
    return DEFAULT_CONFIGS


def run_comparison(
    configs: list[dict[str, Any]],
    image_paths: list[Path],
    output_dir: Path,
    save_images: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run all configurations over identical image paths and compile comparative metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    # Map: image_path -> {config_name: vehicle_count}
    per_image_map: dict[str, dict[str, Any]] = {str(p): {"image_path": str(p)} for p in image_paths}

    for cfg in configs:
        cfg_name = cfg["name"]
        model_name = cfg.get("model", "yolo11n.pt")
        conf_val = float(cfg.get("conf", 0.25))
        imgsz_val = cfg.get("imgsz", None)
        device_val = cfg.get("device", None)

        print(f"\n--- Running configuration: {cfg_name} (model={model_name}, conf={conf_val}) ---")

        detector = VehicleDetector(
            model_path=model_name,
            conf=conf_val,
            imgsz=imgsz_val,
            device=device_val,
        )

        cfg_img_out = (output_dir / "visualizations" / cfg_name) if save_images else None

        records, summary = run_batch_detection(
            detector=detector,
            image_paths=image_paths,
            output_dir=cfg_img_out,
            save_annotated=save_images,
        )

        summary_rows.append({
            "config_name": cfg_name,
            "model": model_name,
            "conf": conf_val,
            "imgsz": imgsz_val if imgsz_val else "default",
            "total_images": summary["total_images"],
            "successful_images": summary["successful_images"],
            "total_vehicles": summary["total_vehicles_detected"],
            "avg_inference_ms": summary["avg_inference_time_ms"],
            "total_time_ms": summary["total_inference_time_ms"],
            "description": cfg.get("description", ""),
        })

        for r in records:
            img_key = r["image_path"]
            if img_key in per_image_map:
                per_image_map[img_key][f"{cfg_name}_vehicles"] = r["vehicle_count"]
                per_image_map[img_key][f"{cfg_name}_ms"] = r["inference_ms"]

    per_image_rows = list(per_image_map.values())

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "num_images_compared": len(image_paths),
    }

    return summary_rows, per_image_rows, metadata


def export_comparison(
    summary_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary CSV
    summary_csv = output_dir / "comparison_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # 2. Per-image comparison CSV
    per_image_csv = output_dir / "per_image_comparison.csv"
    if per_image_rows:
        fields = list(per_image_rows[0].keys())
        with per_image_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_image_rows)

    # 3. Complete JSON report
    report_json = output_dir / "comparison_report.json"
    report_data = {
        "metadata": metadata,
        "configurations_summary": summary_rows,
        "per_image_details": per_image_rows,
    }
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return summary_csv, per_image_csv, report_json


def main() -> None:
    args = parse_args()

    try:
        configs = load_configurations(args.configs_json)
        image_paths = collect_image_paths(args.source)
    except Exception as exc:
        print(f"Error during initialization: {exc}", file=sys.stderr)
        sys.exit(1)

    if not image_paths:
        print(f"No images found in: {args.source}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / f"comparison_{timestamp}"

    print(f"Starting comparison of {len(configs)} configurations across {len(image_paths)} images.")

    summary_rows, per_image_rows, metadata = run_comparison(
        configs=configs,
        image_paths=image_paths,
        output_dir=output_dir,
        save_images=args.save_images,
    )

    summary_csv, per_image_csv, report_json = export_comparison(
        summary_rows=summary_rows,
        per_image_rows=per_image_rows,
        metadata=metadata,
        output_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print("YOLO CONFIGURATION BENCHMARK COMPARISON")
    print("=" * 80)
    header = f"{'Configuration':<20} | {'Model':<12} | {'Conf':<6} | {'Vehicles':<9} | {'Avg ms':<8} | {'Total ms':<9}"
    print(header)
    print("-" * 80)
    for r in summary_rows:
        line = (
            f"{r['config_name']:<20} | "
            f"{r['model']:<12} | "
            f"{r['conf']:<6.2f} | "
            f"{r['total_vehicles']:<9} | "
            f"{r['avg_inference_ms']:<8.1f} | "
            f"{r['total_time_ms']:<9.1f}"
        )
        print(line)
    print("=" * 80)
    print(f"Comparison summary CSV:    {summary_csv}")
    print(f"Per-image comparison CSV:  {per_image_csv}")
    print(f"Full JSON report:          {report_json}")
    if args.save_images:
        print(f"Visualizations saved to:   {output_dir / 'visualizations'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
