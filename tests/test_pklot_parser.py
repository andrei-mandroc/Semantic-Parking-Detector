from pathlib import Path

import cv2
import pytest

from src.data.pklot_parser import parse_annotation


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
SUBSET_DIRS = sorted(
    path for path in DATASET_ROOT.rglob("2013-04-14") if path.is_dir()
)
XML_PATHS = sorted(
    xml_path
    for subset_dir in SUBSET_DIRS
    for xml_path in subset_dir.glob("*.xml")
)

if not XML_PATHS:
    pytest.skip("Local PKLot 2013-04-14 subset is unavailable", allow_module_level=True)


def test_real_subset_has_at_least_ten_annotations():
    assert len(XML_PATHS) >= 10


@pytest.mark.parametrize("xml_path", XML_PATHS, ids=lambda path: path.stem)
def test_real_annotation_matches_image(xml_path: Path):
    image_path = xml_path.with_suffix(".jpg")
    image = cv2.imread(str(image_path))
    assert image is not None, f"Missing or unreadable image: {image_path}"

    spaces, ground_truth = parse_annotation(xml_path)
    space_ids = [space["id"] for space in spaces]
    assert spaces, f"No parking spaces in {xml_path}"
    assert len(space_ids) == len(set(space_ids))
    assert set(ground_truth) == set(space_ids)
    assert all("occupied" not in space for space in spaces)

    height, width = image.shape[:2]
    for space in spaces:
        assert len(space["polygon"]) >= 3
        assert all(
            0 <= x < width and 0 <= y < height
            for x, y in space["polygon"]
        ), f"Space {space['id']} is outside {image_path}"
