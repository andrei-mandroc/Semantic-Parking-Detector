import sys

import cv2
import numpy as np
import pytest

from scripts.visualize_annotations import main
from src.visualization.annotations import render_annotations


SPACE = {"id": "1", "polygon": [[20, 20], [50, 20], [50, 50], [20, 50]]}


def test_geometry_mode_ignores_ground_truth():
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    occupied = render_annotations(image, [SPACE], {"1": True}, "geometry")
    free = render_annotations(image, [SPACE], {"1": False}, "geometry")

    assert np.array_equal(occupied, free)
    assert occupied[20, 35].tolist() == [255, 255, 0]
    assert not image.any()


@pytest.mark.parametrize(
    ("ground_truth", "expected_color"),
    [({"1": True}, [0, 0, 255]), ({"1": False}, [0, 180, 0]), ({}, [160, 160, 160])],
)
def test_ground_truth_mode_colors_contour(ground_truth, expected_color):
    image = np.zeros((80, 80, 3), dtype=np.uint8)

    result = render_annotations(image, [SPACE], ground_truth, "ground-truth")

    assert result[20, 35].tolist() == expected_color


def test_polygon_outside_image_raises():
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside image bounds"):
        render_annotations(image, [SPACE], {}, "geometry")


def test_cli_saves_visualization(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.jpg"
    xml_path = tmp_path / "sample.xml"
    output_path = tmp_path / "annotated.png"
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    xml_path.write_text(
        '<parking><space id="1" occupied="1"><contour>'
        '<point x="20" y="20" /><point x="50" y="20" />'
        '<point x="50" y="50" /><point x="20" y="50" />'
        '</contour></space></parking>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_annotations",
            "--image", str(image_path),
            "--xml", str(xml_path),
            "--mode", "ground-truth",
            "--output", str(output_path),
        ],
    )

    main()

    annotated = cv2.imread(str(output_path))
    assert annotated is not None
    assert annotated.shape == image.shape
    assert annotated[20, 35].tolist() == [0, 0, 255]


def test_cli_rejects_mismatched_image_and_xml(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "first.jpg"
    xml_path = tmp_path / "second.xml"
    image_path.write_bytes(b"image")
    xml_path.write_text("<parking />", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["visualize_annotations", "--image", str(image_path), "--xml", str(xml_path)],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "filenames do not match" in capsys.readouterr().err


def test_cli_does_not_overwrite_input_image(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "sample.jpg"
    xml_path = tmp_path / "sample.xml"
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    original_bytes = image_path.read_bytes()
    xml_path.write_text(
        '<parking><space id="1" occupied="0"><contour>'
        '<point x="20" y="20" /><point x="50" y="20" />'
        '<point x="50" y="50" /></contour></space></parking>',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_annotations",
            "--image", str(image_path),
            "--xml", str(xml_path),
            "--output", str(image_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "must differ from the input image" in capsys.readouterr().err
    assert image_path.read_bytes() == original_bytes
