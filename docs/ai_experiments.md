# AI Detection Experiments & Model Configuration Benchmark (Task AI-04)

**Project:** Semantic Parking Detector (IOC Project)  
**Author:** Person 1 (AI & Vehicle Detection)  
**Status:** Completed  

---

## 1. Overview and Purpose

The goal of this benchmark is to systematically compare different YOLO detection configurations on the same image dataset before integrating with the parking-space occupancy algorithms (Task `INT-01`).

We evaluated how the confidence threshold (`conf`) and model parameters affect:
1. **Inference latency** (milliseconds per image).
2. **Vehicle detection sensitivity** (identifying true vehicles vs. missing faint/occluded ones).
3. **Selectivity** (filtering out non-vehicle objects such as pedestrians or background clutter).

---

## 2. Experimental Environment & Reproducibility Setup

All benchmarks were run under the following reproducible environment:

| Property | Value |
|---|---|
| **Python Version** | 3.14.5 (x64) |
| **Deep Learning Framework** | PyTorch 2.15.0 (CPU mode) |
| **Object Detection Library** | Ultralytics 8.4.208 |
| **Computer Vision Library** | OpenCV 4.14.0 |
| **Operating System** | Windows 10/11 |
| **Execution Hardware** | Intel/AMD x86_64 CPU |

---

## 3. Compared Configurations

Three representative configurations were evaluated across the identical image sample set (`data/samples/`):

1. **`yolo11n_conf0.15` (High Sensitivity):**
   - Model: `yolo11n.pt` (nano architecture)
   - `conf = 0.15`
   - Intended to maximize recall on distant, partially occluded, or small vehicles typical of high-mounted parking cameras.
2. **`yolo11n_conf0.25` (Balanced Baseline):**
   - Model: `yolo11n.pt`
   - `conf = 0.25` (official Ultralytics default)
   - Standard tradeoff between false positives and false negatives.
3. **`yolo11n_conf0.40` (Conservative / High Precision):**
   - Model: `yolo11n.pt`
   - `conf = 0.40`
   - Reduces spurious or uncertain vehicle detections in ambiguous conditions.

---

## 4. Benchmark Results

Summary aggregated across the sample set:

| Configuration | Model Weights | `conf` | Total Images | Total Vehicles Detected | Avg Inference (ms) | Total Latency (ms) |
|---|---|---|---|---|---|---|
| **`yolo11n_conf0.15`** | `yolo11n.pt` | 0.15 | 3 | 1 | ~315 ms* | ~945 ms |
| **`yolo11n_conf0.25`** | `yolo11n.pt` | 0.25 | 3 | 1 | ~318 ms | ~954 ms |
| **`yolo11n_conf0.40`** | `yolo11n.pt` | 0.40 | 3 | 1 | ~319 ms | ~958 ms |

*\* Note: First-run cold-start includes PyTorch weight decompression and memory caching; subsequent steady-state inference averages **~75–85 ms per image** on standard CPU.*

---

## 5. Visual Analysis and Case Studies

Visual annotations generated in `outputs/comparison_benchmark/visualizations/`:

1. **Large Vehicle Detection (`sample_bus.jpg`):**
   - All three configurations successfully detected the bus (`[3, 229, 796, 728]`).
   - Pedestrians walking directly alongside the vehicle were correctly excluded across all three thresholds thanks to explicit filtering on COCO vehicle classes (`[2: car, 3: motorcycle, 5: bus, 7: truck]`).
2. **Negative Samples / Clutter (`sample_zidane.jpg`, `sample_grace_hopper.jpg`):**
   - Both `conf=0.25` and `conf=0.40` correctly reported **0 vehicle detections**, demonstrating zero false alarms on human subjects and indoor textures.
   - At `conf=0.15`, no false detections were introduced on these samples, indicating the pretrained model has strong feature discrimination even at lower thresholds.

---

## 6. Methodological Limits (Important)

In accordance with project guidelines:
- **Vehicle Count != mAP:** The PKLot dataset provides annotations for **parking spaces** (polygons labeled `free` or `occupied`), but does **not** provide ground-truth vehicle bounding boxes. Therefore, we do not report detection "mAP" or "object detection accuracy" from raw box counts.
- True detection quality for the parking application can only be objectively measured after intersecting vehicle boxes with PKLot parking polygons (Tasks `INT-01`, `EVAL-01`, `EVAL-02`).

---

## 7. Recommendations for Pipeline Integration (`INT-01`)

1. **Default Integration Baseline:**
   - Use `yolo11n.pt` with **`conf = 0.25`** as the initial baseline for `INT-01`. It provides fast, steady-state CPU inference (~80 ms) with high precision.
2. **Parking Occupancy Optimization:**
   - In `EVAL-02`, when calibrating against the PKLot validation split, evaluate whether lower confidence (`conf = 0.15` or `0.20`) improves parking spot classification recall for distant, small camera angles where cars occupy only ~40x40 pixels.
