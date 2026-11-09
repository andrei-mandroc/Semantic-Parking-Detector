"""Tests for vehicle detection and visualization (AI-01)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.visualization.render import render_detections


def test_render_detections_empty():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = render_detections(image, [])
    assert rendered.shape == image.shape
    assert np.array_equal(rendered, image)


def test_render_detections_draws_boxes():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [(10, 10, 50, 50), (60, 60, 90, 90)]
    rendered = render_detections(image, boxes)
    assert rendered.shape == image.shape
    # Rendered image must have non-zero pixels where rectangles/labels were drawn
    assert np.any(rendered != 0)


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_with_boxes(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    # Mock xyxy tensor
    mock_boxes = MagicMock()
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.2, 20.7, 50.1, 80.9],
        [100.0, 120.0, 180.0, 210.0],
    ])
    mock_boxes.xyxy = mock_xyxy
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    from src.detection.yolo_detector import VehicleDetector

    detector = VehicleDetector(model_path="dummy.pt")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.predict(dummy_img)

    assert len(detections) == 2
    assert detections[0] == (10, 20, 50, 80)
    assert detections[1] == (100, 120, 180, 210)
    assert all(isinstance(val, int) for box in detections for val in box)


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_no_boxes(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.predict.return_value = [mock_result]

    from src.detection.yolo_detector import VehicleDetector

    detector = VehicleDetector(model_path="dummy.pt")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.predict(dummy_img)

    assert detections == []
