# Project Scripts

Utility and experiment scripts for the Semantic Parking Detector.

## Scripts Overview

| Script | Purpose | Usage Example |
|---|---|---|
| `scripts/test_detection.py` | Smoke test vehicle detection on a single image (Task AI-01) | `python scripts/test_detection.py --image data/samples/sample_bus.jpg` |

## Options for `test_detection.py`

- `--image` (required): Path to the input image file.
- `--model` (optional, default: `yolo11n.pt`): Model path or weights name.
- `--output` (optional): Destination path for annotated output image (defaults to `outputs/<name>_detection.jpg`).

## Smoke Test Results (Task AI-01)

| Input Image | Resolution | Ground Truth Content | Detected Vehicles | Inference Time | Status |
|---|---|---|---|---|---|
| `data/samples/sample_bus.jpg` | 810x1080 | 1 Bus, multiple pedestrians | 1 `[3, 229, 796, 728]` (Bus) | ~1.6s (CPU) | **Passed** (Correctly filtered out people, detected bus) |
| `data/samples/sample_zidane.jpg` | 1280x720 | 2 Persons, no vehicles | 0 | ~1.5s (CPU) | **Passed** (Correctly identified 0 vehicles, no false positives) |
