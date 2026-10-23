from src.parking.occupancy import classify_spaces


SPACES = [{"id": "A1", "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]]}]


def test_no_detections_means_free():
    assert classify_spaces((100, 100), SPACES, []) == {"A1": False}


def test_vehicle_on_space_means_occupied():
    assert classify_spaces((100, 100), SPACES, [(15, 15, 45, 45)]) == {"A1": True}


def test_vehicle_outside_space_means_free():
    assert classify_spaces((100, 100), SPACES, [(65, 65, 80, 80)]) == {"A1": False}
