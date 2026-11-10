"""Unit tests for detector configuration comparison (AI-04)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.compare_detectors import export_comparison, load_configurations, run_comparison


def test_load_configurations_default():
    configs = load_configurations(None)
    assert len(configs) >= 2
    assert all("name" in c and "conf" in c for c in configs)


def test_load_configurations_custom(tmp_path: Path):
    cfg_file = tmp_path / "custom_configs.json"
    custom_data = [
        {"name": "test_cfg", "model": "yolo11n.pt", "conf": 0.35}
    ]
    cfg_file.write_text(json.dumps(custom_data), encoding="utf-8")

    loaded = load_configurations(cfg_file)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "test_cfg"
    assert loaded[0]["conf"] == 0.35


def test_load_configurations_missing_file():
    with pytest.raises(FileNotFoundError):
        load_configurations(Path("nonexistent_configs_file.json"))


@patch("scripts.compare_detectors.run_batch_detection")
@patch("scripts.compare_detectors.VehicleDetector")
def test_run_comparison_mocked(mock_detector_cls, mock_run_batch, tmp_path: Path):
    mock_run_batch.side_effect = [
        # Result for Config 1
        (
            [{"image_path": "img1.jpg", "vehicle_count": 2, "inference_ms": 10.0}],
            {"total_images": 1, "successful_images": 1, "total_vehicles_detected": 2, "avg_inference_time_ms": 10.0, "total_inference_time_ms": 10.0},
        ),
        # Result for Config 2
        (
            [{"image_path": "img1.jpg", "vehicle_count": 1, "inference_ms": 11.0}],
            {"total_images": 1, "successful_images": 1, "total_vehicles_detected": 1, "avg_inference_time_ms": 11.0, "total_inference_time_ms": 11.0},
        ),
    ]

    configs = [
        {"name": "cfg1", "model": "yolo11n.pt", "conf": 0.25},
        {"name": "cfg2", "model": "yolo11n.pt", "conf": 0.45},
    ]
    img_paths = [Path("img1.jpg")]

    summary, per_img, metadata = run_comparison(
        configs=configs,
        image_paths=img_paths,
        output_dir=tmp_path / "comparison_out",
        save_images=False,
    )

    assert len(summary) == 2
    assert summary[0]["config_name"] == "cfg1"
    assert summary[0]["total_vehicles"] == 2
    assert summary[1]["config_name"] == "cfg2"
    assert summary[1]["total_vehicles"] == 1

    assert len(per_img) == 1
    assert per_img[0]["cfg1_vehicles"] == 2
    assert per_img[0]["cfg2_vehicles"] == 1


def test_export_comparison(tmp_path: Path):
    summary_rows = [
        {"config_name": "c1", "model": "m.pt", "conf": 0.25, "total_vehicles": 5}
    ]
    per_image_rows = [
        {"image_path": "img1.jpg", "c1_vehicles": 5}
    ]
    metadata = {"platform": "test"}

    out_dir = tmp_path / "export_test"
    csv_path, per_img_csv, json_path = export_comparison(
        summary_rows, per_image_rows, metadata, out_dir
    )

    assert csv_path.exists()
    assert per_img_csv.exists()
    assert json_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        assert "metadata" in data
        assert "configurations_summary" in data
