"""Unit tests for batch detection functionality (AI-03)."""
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.detection.batch import collect_image_paths, export_batch_results, run_batch_detection


@pytest.fixture
def temp_image_dir(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i}.jpg"), img)

    # Add a non-image file that should be ignored
    (img_dir / "ignored.txt").write_text("hello", encoding="utf-8")
    return img_dir


def test_collect_image_paths_from_dir(temp_image_dir: Path):
    paths = collect_image_paths(temp_image_dir)
    assert len(paths) == 3
    assert all(p.suffix == ".jpg" for p in paths)


def test_collect_image_paths_from_csv(tmp_path: Path):
    csv_file = tmp_path / "manifest.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "weather"])
        writer.writerow(["data/img1.jpg", "sunny"])
        writer.writerow(["data/img2.png", "rainy"])

    paths = collect_image_paths(csv_file)
    assert len(paths) == 2
    assert paths[0] == Path("data/img1.jpg")
    assert paths[1] == Path("data/img2.png")


def test_collect_image_paths_from_txt(tmp_path: Path):
    txt_file = tmp_path / "list.txt"
    txt_file.write_text("data/img1.jpg\n# comment\ndata/img2.jpg\n\n", encoding="utf-8")

    paths = collect_image_paths(txt_file)
    assert len(paths) == 2
    assert paths[0] == Path("data/img1.jpg")
    assert paths[1] == Path("data/img2.jpg")


def test_collect_image_paths_nonexistent():
    with pytest.raises(FileNotFoundError):
        collect_image_paths("nonexistent_directory_12345")


def test_run_batch_detection_mocked(temp_image_dir: Path, tmp_path: Path):
    images = collect_image_paths(temp_image_dir)

    mock_detector = MagicMock()
    mock_detector.model_path = "mock_model.pt"
    mock_detector.conf = 0.3
    mock_detector.predict_detailed.return_value = [
        {"box": (10, 10, 40, 40), "confidence": 0.85, "class_id": 2}
    ]

    out_dir = tmp_path / "output"
    records, summary = run_batch_detection(
        detector=mock_detector,
        image_paths=images,
        output_dir=out_dir,
        save_annotated=True,
    )

    assert summary["total_images"] == 3
    assert summary["successful_images"] == 3
    assert summary["failed_images"] == 0
    assert summary["total_vehicles_detected"] == 3
    assert len(records) == 3

    for r in records:
        assert r["status"] == "ok"
        assert r["vehicle_count"] == 1
        assert r["image_width"] == 50
        assert r["image_height"] == 50

    # Verify annotated images were saved
    saved_images = list(out_dir.glob("*_detected.jpg"))
    assert len(saved_images) == 3


def test_run_batch_detection_handles_missing_file(tmp_path: Path):
    mock_detector = MagicMock()
    mock_detector.model_path = "mock.pt"
    mock_detector.conf = 0.25

    fake_paths = [tmp_path / "does_not_exist.jpg"]
    records, summary = run_batch_detection(
        detector=mock_detector,
        image_paths=fake_paths,
    )

    assert summary["total_images"] == 1
    assert summary["successful_images"] == 0
    assert summary["failed_images"] == 1
    assert records[0]["status"] == "error"
    assert records[0]["error_message"] == "File not found"


def test_export_batch_results(tmp_path: Path):
    records = [
        {
            "image_path": "a.jpg",
            "model": "m.pt",
            "confidence_threshold": 0.25,
            "image_width": 100,
            "image_height": 100,
            "vehicle_count": 2,
            "inference_ms": 15.2,
            "status": "ok",
            "error_message": "",
            "boxes": [(10, 10, 50, 50), (60, 60, 90, 90)],
        }
    ]
    summary = {
        "model": "m.pt",
        "total_images": 1,
        "successful_images": 1,
        "failed_images": 0,
        "total_vehicles_detected": 2,
    }

    csv_path, json_path = export_batch_results(records, summary, tmp_path)

    assert csv_path.exists()
    assert json_path is not None and json_path.exists()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["vehicle_count"] == "2"

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["summary"]["total_images"] == 1
        assert len(data["records"]) == 1
