from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.manifest import assign_splits, build_manifest


def _pair(root: Path, directory: str, name: str, xml: str | None = None) -> None:
    folder = root / directory
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{name}.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
    (folder / f"{name}.xml").write_text(
        xml or '<annotation><space id="1" occupied="0"><contour>'
        '<point x="1" y="1"/><point x="10" y="1"/>'
        '<point x="10" y="10"/></contour></space></annotation>',
        encoding="utf-8",
    )


def test_manifest_uses_relative_paths_and_infers_metadata(tmp_path: Path):
    _pair(tmp_path, "UFPR05/Cloudy/2013-04-14", "2013-04-14_06_40_00")

    rows, report = build_manifest(tmp_path)

    assert len(rows) == 1
    assert rows[0] == {
        "manifest_version": "1",
        "image_path": "UFPR05/Cloudy/2013-04-14/2013-04-14_06_40_00.jpg",
        "xml_path": "UFPR05/Cloudy/2013-04-14/2013-04-14_06_40_00.xml",
        "file_name": "2013-04-14_06_40_00.jpg",
        "camera": "UFPR05",
        "weather": "cloudy",
        "date": "2013-04-14",
        "session": "2013-04-14",
        "split": "train",
    }
    assert report["by_camera"] == {"UFPR05": 1}
    assert report["by_weather"] == {"cloudy": 1}
    assert report["by_date"] == {"2013-04-14": 1}
    assert report["by_split"] == {"train": 1, "validation": 0, "test": 0}


def test_manifest_reports_orphans_and_invalid_pairs(tmp_path: Path):
    _pair(tmp_path, "day", "valid")
    _pair(tmp_path, "day", "bad_xml", "<annotation>")
    _pair(
        tmp_path, "day", "out_of_bounds",
        '<annotation><space id="1" occupied="1"><contour>'
        '<point x="1" y="1"/><point x="25" y="1"/>'
        '<point x="10" y="10"/></contour></space></annotation>',
    )
    (tmp_path / "day" / "image_only.jpg").write_bytes(b"image")
    (tmp_path / "day" / "xml_only.xml").write_text("<annotation/>", encoding="utf-8")

    rows, report = build_manifest(tmp_path)

    assert len(rows) == 1
    assert report["images_without_xml"] == ["day/image_only.jpg"]
    assert report["xml_without_images"] == ["day/xml_only.xml"]
    assert {item["xml_path"] for item in report["invalid_pairs"]} == {
        "day/bad_xml.xml", "day/out_of_bounds.xml"
    }


def test_splits_are_reproducible_and_days_do_not_overlap(tmp_path: Path):
    for day in range(1, 11):
        date = f"2013-04-{day:02d}"
        for camera in ("UFPR04", "UFPR05"):
            _pair(tmp_path, f"{camera}/Sunny/{date}", f"{date}_06_40_00")

    rows, report = build_manifest(tmp_path, seed=7)
    repeated_rows, repeated_report = build_manifest(tmp_path, seed=7)

    assert rows == repeated_rows
    assert report == repeated_report
    assert set(report["by_split"]) == {"train", "validation", "test"}
    assert all(report["by_split"].values())
    for date in {row["date"] for row in rows}:
        assert len({row["split"] for row in rows if row["date"] == date}) == 1


def test_sessions_without_dates_are_grouped_by_directory():
    rows = [
        {"session": "directory:camera/day_a"},
        {"session": "directory:camera/day_a"},
        {"session": "directory:camera/day_b"},
    ]
    assigned = assign_splits(rows)
    assert assigned[0]["split"] == assigned[1]["split"]
    assert assigned[0]["split"] != assigned[2]["split"]


def test_missing_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="Dataset directory does not exist"):
        build_manifest(tmp_path / "missing")


def test_local_pklot_days_have_valid_manifest_and_disjoint_splits():
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "PKLot"
    if not root.is_dir():
        pytest.skip("Local PKLot subset is unavailable")

    rows, report = build_manifest(root)

    expected_days = {"2013-04-12": 18, "2013-04-13": 106, "2013-04-14": 56}
    assert len(rows) >= sum(expected_days.values())
    assert report["valid_pairs"] == len(rows)
    assert all(report["by_date"][day] == count for day, count in expected_days.items())
    assert report["by_camera"] == {"UFPR05": len(rows)}
    assert report["by_weather"] == {"cloudy": len(rows)}
    assert all(report["by_split"].values())
    for day in report["by_date"]:
        assert len({row["split"] for row in rows if row["date"] == day}) == 1
    assert report["spaces_without_valid_labels"] == 9
    assert not report["invalid_pairs"]
    assert not report["images_without_xml"]
    assert not report["xml_without_images"]
