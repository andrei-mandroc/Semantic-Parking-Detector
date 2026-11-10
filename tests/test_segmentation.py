"""Unit tests for vehicle instance segmentation (AI-05)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.segmentation_detector import (
    VehicleSegmentationDetector,
    render_segmentation_overlay,
)


def test_render_segmentation_overlay_empty():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rendered = render_segmentation_overlay(img, [])
    assert rendered.shape == img.shape


def test_render_segmentation_overlay_draws_polygons():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    poly = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
    detections = [
        {"box": (20, 20, 80, 80), "polygon": poly, "confidence": 0.95, "class_id": 2}
    ]
    rendered = render_segmentation_overlay(img, detections)
    assert rendered.shape == img.shape
    assert np.any(rendered != 0)


@patch("src.detection.segmentation_detector.YOLO")
def test_segmentation_detector_predict_masks(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_masks = MagicMock()
    mock_masks.xy = [
        np.array([[10, 10], [50, 10], [50, 50], [10, 50]]),
    ]
    mock_result.masks = mock_masks
    mock_model.predict.return_value = [mock_result]

    detector = VehicleSegmentationDetector(model_path="dummy-seg.pt")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    masks = detector.predict_masks(img)

    assert len(masks) == 1
    assert len(masks[0]) == 4


@patch("src.detection.segmentation_detector.YOLO")
def test_segmentation_detector_predict_detailed(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([2])
    mock_result.boxes = mock_boxes

    mock_masks = MagicMock()
    mock_masks.xy = [
        np.array([[10, 10], [50, 10], [50, 50], [10, 50]]),
    ]
    mock_result.masks = mock_masks
    mock_model.predict.return_value = [mock_result]

    detector = VehicleSegmentationDetector()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    detailed = detector.predict_detailed(img)

    assert len(detailed) == 1
    assert detailed[0]["box"] == (10, 10, 50, 50)
    assert detailed[0]["polygon"] is not None
    assert len(detailed[0]["polygon"]) == 4
    assert detailed[0]["class_id"] == 2
    assert abs(detailed[0]["confidence"] - 0.92) < 1e-4


@patch("src.detection.segmentation_detector.YOLO")
def test_segmentation_detector_predict_backward_compat(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[15, 25, 60, 70]])
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    detector = VehicleSegmentationDetector()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = detector.predict(img)

    assert boxes == [(15, 25, 60, 70)]


@patch("src.detection.segmentation_detector.YOLO")
def test_segmentation_detector_no_detections(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_result.boxes = None
    mock_result.masks = None
    mock_model.predict.return_value = [mock_result]

    detector = VehicleSegmentationDetector()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert detector.predict(img) == []
    assert detector.predict_masks(img) == []
    assert detector.predict_detailed(img) == []
