import pytest

from src.parking.occupancy import (
    classify_spaces,
    classify_spaces_by_center,
    coverage_scores,
)


SPACES = [{"id": "A1", "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]]}]


def test_no_detections_means_free():
    assert classify_spaces((100, 100), SPACES, []) == {"A1": False}


def test_vehicle_on_space_means_occupied():
    assert classify_spaces((100, 100), SPACES, [(15, 15, 45, 45)]) == {"A1": True}


def test_vehicle_outside_space_means_free():
    assert classify_spaces((100, 100), SPACES, [(65, 65, 80, 80)]) == {"A1": False}


def test_partial_overlap_respects_threshold():
    boxes = [(45, 20, 55, 40)]

    assert coverage_scores((100, 100), SPACES, boxes)["A1"] == pytest.approx(6 * 20 / 41**2)
    assert classify_spaces((100, 100), SPACES, boxes, threshold=0.05) == {"A1": True}
    assert classify_spaces((100, 100), SPACES, boxes, threshold=0.1) == {"A1": False}


def test_duplicate_boxes_do_not_double_count_coverage():
    box = (15, 15, 45, 45)

    assert coverage_scores((100, 100), SPACES, [box, box]) == coverage_scores(
        (100, 100), SPACES, [box]
    )


def test_negative_box_coordinates_are_clipped():
    box = (-10, -10, 25, 25)

    assert coverage_scores((100, 100), SPACES, [box])["A1"] == pytest.approx(15 * 15 / 41**2)
    assert classify_spaces_by_center((100, 100), SPACES, [box]) == {"A1": True}


def test_multiple_spaces_are_classified_independently():
    spaces = SPACES + [
        {"id": "A2", "polygon": [[60, 10], [90, 10], [90, 40], [60, 40]]}
    ]
    box = (15, 15, 45, 45)

    assert classify_spaces((100, 100), spaces, [box]) == {"A1": True, "A2": False}
    assert classify_spaces_by_center((100, 100), spaces, [box]) == {
        "A1": True,
        "A2": False,
    }


def test_small_image_and_box_outside_image():
    spaces = [{"id": "tiny", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}]

    assert classify_spaces((3, 3), spaces, [(0, 0, 1, 1)], threshold=0.1) == {
        "tiny": True
    }
    assert classify_spaces((3, 3), spaces, [(5, 5, 8, 8)]) == {"tiny": False}


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan")])
def test_invalid_threshold_raises(threshold):
    with pytest.raises(ValueError, match="threshold"):
        classify_spaces((100, 100), SPACES, [], threshold=threshold)


@pytest.mark.parametrize(
    ("spaces", "error"),
    [
        ([{"id": "bad", "polygon": [[10, 10], [20, 20]]}], "at least 3"),
        ([{"id": "bad", "polygon": [[10, 10], [20, 20], [30, 30]]}], "zero-area"),
        ([{"id": "bad", "polygon": [[10, 10], [20, 10], [100, 20]]}], "outside"),
        ([{"id": "bad", "polygon": [[10, 10], [20, 10], [20, float("inf")]]}], "non-finite"),
    ],
)
def test_invalid_polygon_raises(spaces, error):
    with pytest.raises(ValueError, match=error):
        classify_spaces((100, 100), spaces, [])
    with pytest.raises(ValueError, match=error):
        classify_spaces_by_center((100, 100), spaces, [])


def test_duplicate_space_ids_raise():
    with pytest.raises(ValueError, match="unique"):
        classify_spaces((100, 100), SPACES + SPACES, [])


def test_invalid_image_dimensions_raise():
    with pytest.raises(ValueError, match="dimensions"):
        classify_spaces_by_center((0, 100), SPACES, [])


def test_center_rule_uses_box_center_instead_of_coverage():
    space = [{"id": "A1", "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]]}]
    box = (25, 15, 45, 25)

    assert classify_spaces((50, 50), space, [box], threshold=0.1) == {"A1": True}
    assert classify_spaces_by_center((50, 50), space, [box]) == {"A1": False}


def test_center_rule_treats_polygon_boundary_as_inside():
    space = [{"id": "A1", "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]]}]

    assert classify_spaces_by_center((50, 50), space, [(0, 0, 20, 20)]) == {"A1": True}
    assert classify_spaces_by_center((50, 50), space, []) == {"A1": False}
