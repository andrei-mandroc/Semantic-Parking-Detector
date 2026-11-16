"""Tests for vehicle detection and visualization (AI-01 & AI-02)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detection.yolo_detector import DEFAULT_VEHICLE_CLASSES, VehicleDetector
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
    assert np.any(rendered != 0)


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_initialization_defaults(mock_yolo_cls):
    detector = VehicleDetector()
    mock_yolo_cls.assert_called_once_with("yolo11n.pt")
    assert detector.conf == 0.25
    assert detector.imgsz is None
    assert detector.classes == DEFAULT_VEHICLE_CLASSES


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_custom_parameters(mock_yolo_cls):
    detector = VehicleDetector(
        model_path="custom.pt",
        conf=0.45,
        imgsz=1280,
        classes=[2, 7],
        device="cpu",
    )
    mock_yolo_cls.assert_called_once_with("custom.pt")
    assert detector.conf == 0.45
    assert detector.imgsz == 1280
    assert detector.classes == [2, 7]
    assert detector.device == "cpu"


@patch("src.detection.yolo_detector.YOLO")
@pytest.mark.parametrize("invalid_conf", [0.0, -0.1, 1.5, "0.5", True, False, None])
def test_vehicle_detector_invalid_conf(mock_yolo_cls, invalid_conf):
    with pytest.raises(ValueError):
        VehicleDetector(conf=invalid_conf)


@patch("src.detection.yolo_detector.YOLO")
@pytest.mark.parametrize("invalid_imgsz", [0, -640, "640", (640,), (640, -480), True])
def test_vehicle_detector_invalid_imgsz(mock_yolo_cls, invalid_imgsz):
    with pytest.raises(ValueError):
        VehicleDetector(imgsz=invalid_imgsz)


@patch("src.detection.yolo_detector.YOLO")
@pytest.mark.parametrize("invalid_classes", ["2", [-1], [2, "car"], 2])
def test_vehicle_detector_invalid_classes(mock_yolo_cls, invalid_classes):
    with pytest.raises((ValueError, TypeError)):
        VehicleDetector(classes=invalid_classes)


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_with_boxes(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([
        [10.2, 20.7, 50.1, 80.9],
        [100.0, 120.0, 180.0, 210.0],
    ])
    mock_boxes.xyxy = mock_xyxy
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    detector = VehicleDetector(model_path="dummy.pt", conf=0.3)
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.predict(dummy_img)

    assert len(detections) == 2
    assert detections[0] == (10, 20, 50, 80)
    assert detections[1] == (100, 120, 180, 210)
    assert all(isinstance(val, int) for box in detections for val in box)
    mock_model.predict.assert_called_once_with(
        source=dummy_img,
        classes=DEFAULT_VEHICLE_CLASSES,
        conf=0.3,
        verbose=False,
    )


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_conf_override(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.predict.return_value = [mock_result]

    detector = VehicleDetector(conf=0.25)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.predict(dummy_img, conf=0.6)

    mock_model.predict.assert_called_once_with(
        source=dummy_img,
        classes=DEFAULT_VEHICLE_CLASSES,
        conf=0.6,
        verbose=False,
    )


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_no_boxes(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.predict.return_value = [mock_result]

    detector = VehicleDetector(model_path="dummy.pt")
    dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.predict(dummy_img)

    assert detections == []


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_single_box(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_xyxy = MagicMock()
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([[5, 10, 50, 60]])
    mock_boxes.xyxy = mock_xyxy
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    detector = VehicleDetector()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.predict(dummy_img)

    assert detections == [(5, 10, 50, 60)]


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_predict_detailed(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    mock_result = MagicMock()
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 30, 40]])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.88])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([2])
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    detector = VehicleDetector()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    detailed = detector.predict_detailed(dummy_img)

    assert len(detailed) == 1
    assert detailed[0]["box"] == (10, 20, 30, 40)
    assert abs(detailed[0]["confidence"] - 0.88) < 1e-4
    assert detailed[0]["class_id"] == 2


@patch("src.detection.yolo_detector.YOLO")
def test_vehicle_detector_invalid_image(mock_yolo_cls):
    detector = VehicleDetector()

    with pytest.raises(TypeError):
        detector.predict("not an image")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        detector.predict(np.array([]))
