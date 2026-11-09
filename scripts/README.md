# Project Scripts

Utility and experiment scripts for the Semantic Parking Detector.

## Scripts Overview

| Script | Purpose | Usage Example |
|---|---|---|
| `scripts/test_detection.py` | Smoke test vehicle detection on a single image (Task AI-01) | `python scripts/test_detection.py --image data/samples/sample_bus.jpg` |
| `scripts/detect_batch.py` | Batch vehicle detection on directories or manifests (Task AI-03) | `python scripts/detect_batch.py --source data/samples --save-images` |
| `scripts/compare_detectors.py` | Benchmark and compare multiple YOLO configurations (Task AI-04) | `python scripts/compare_detectors.py --source data/samples --save-images` |
| `scripts/test_segmentation.py` | Instance segmentation test for vehicle contours (Task AI-05) | `python scripts/test_segmentation.py --image data/samples/sample_bus.jpg` |

## Options for `test_detection.py`

- `--image` (required): Path to the input image file.
- `--model` (optional, default: `yolo11n.pt`): Model path or weights name.
- `--conf` (optional, default: `0.25`): Confidence threshold in (0.0, 1.0].
- `--output` (optional): Destination path for annotated output image (defaults to `outputs/<name>_detection.jpg`).

## Options for `detect_batch.py`

- `--source` (required): Path to image directory, CSV manifest, or `.txt` file containing image paths.
- `--model` (optional, default: `yolo11n.pt`): Model path or weights name.
- `--conf` (optional, default: `0.25`): Confidence threshold in (0.0, 1.0].
- `--imgsz` (optional): Input image size for YOLO inference.
- `--device` (optional): Device identifier (`cpu`, `0`).
- `--output-dir` (optional): Destination directory for reports and images (defaults to `outputs/batch_<timestamp>`).
- `--save-images` (optional flag): Saves annotated images with bounding boxes into the output directory.
- `--no-json` (optional flag): Disables generating `detections.json`.

## Options for `compare_detectors.py`

- `--source` (optional, default: `data/samples`): Path to image directory or manifest.
- `--configs-json` (optional): Custom JSON file defining list of configurations.
- `--output-dir` (optional): Destination directory for comparison CSV/JSON reports.
- `--save-images` (optional flag): Saves comparative visual annotations for each configuration.

## Options for `test_segmentation.py`

- `--image` (required): Path to the input image file.
- `--model` (optional, default: `yolo11n-seg.pt`): Model path or weights name.
- `--conf` (optional, default: `0.25`): Confidence threshold in (0.0, 1.0].
- `--output` (optional): Destination path for annotated segmentation output image.

## Smoke Test Results (Task AI-01)

| Input Image | Resolution | Ground Truth Content | Detected Vehicles | Inference Time | Status |
|---|---|---|---|---|---|
| `data/samples/sample_bus.jpg` | 810x1080 | 1 Bus, multiple pedestrians | 1 `[3, 229, 796, 728]` (Bus) | ~1.6s (CPU) | **Passed** (Correctly filtered out people, detected bus) |
| `data/samples/sample_zidane.jpg` | 1280x720 | 2 Persons, no vehicles | 0 | ~1.5s (CPU) | **Passed** (Correctly identified 0 vehicles, no false positives) |
