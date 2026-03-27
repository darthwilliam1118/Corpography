import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pose import PoseDetector, VISIBILITY_THRESHOLD, KEY_LANDMARK_INDICES


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_landmark(visibility: float) -> MagicMock:
    lm = MagicMock()
    lm.visibility = visibility
    return lm


def _make_33_landmarks(default_visibility: float = 1.0) -> list:
    return [_make_landmark(default_visibility) for _ in range(33)]


def _make_result(landmarks_list: list | None) -> MagicMock:
    result = MagicMock()
    if landmarks_list is None:
        result.pose_landmarks = []
    else:
        result.pose_landmarks = [landmarks_list]
    return result


def _patched_open(detector: PoseDetector) -> MagicMock:
    """Open a detector with a mocked landmarker; returns the mock landmarker."""
    mock_landmarker = MagicMock()
    with patch("pose.mp.tasks.vision.PoseLandmarker.create_from_options", return_value=mock_landmarker):
        detector.open()
    return mock_landmarker


# ── construction ──────────────────────────────────────────────────────────────

def test_instantiation_without_model_file():
    detector = PoseDetector(model_path="nonexistent.task")
    assert detector._landmarker is None
    assert detector.is_open() is False


# ── open() ────────────────────────────────────────────────────────────────────

def test_open_returns_false_on_exception():
    detector = PoseDetector(model_path="nonexistent.task")
    with patch("pose.mp.tasks.vision.PoseLandmarker.create_from_options", side_effect=ValueError("missing")):
        result = detector.open()
    assert result is False
    assert detector._landmarker is None


def test_open_returns_true_on_success():
    detector = PoseDetector(model_path="fake.task")
    mock_lm = MagicMock()
    with patch("pose.mp.tasks.vision.PoseLandmarker.create_from_options", return_value=mock_lm):
        result = detector.open()
    assert result is True
    assert detector._landmarker is mock_lm


# ── process() ─────────────────────────────────────────────────────────────────

def test_process_returns_none_when_not_open():
    detector = PoseDetector(model_path="fake.task")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.process(frame, 0) is None


def test_process_calls_detect_for_video():
    detector = PoseDetector(model_path="fake.task")
    mock_lm = MagicMock()
    fake_result = MagicMock()
    mock_lm.detect_for_video.return_value = fake_result

    with patch("pose.mp.tasks.vision.PoseLandmarker.create_from_options", return_value=mock_lm):
        detector.open()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("pose.mp.Image", return_value=MagicMock()):
        result = detector.process(frame, 1234)

    mock_lm.detect_for_video.assert_called_once()
    call_args = mock_lm.detect_for_video.call_args
    assert call_args[0][1] == 1234  # timestamp_ms is second positional arg
    assert result is fake_result


# ── get_landmarks() ───────────────────────────────────────────────────────────

def test_get_landmarks_returns_none_on_none_result():
    detector = PoseDetector(model_path="fake.task")
    assert detector.get_landmarks(None) is None


def test_get_landmarks_returns_none_on_empty_list():
    detector = PoseDetector(model_path="fake.task")
    result = _make_result(None)  # pose_landmarks = []
    assert detector.get_landmarks(result) is None


def test_get_landmarks_returns_first_person():
    detector = PoseDetector(model_path="fake.task")
    lms = _make_33_landmarks()
    result = _make_result(lms)
    assert detector.get_landmarks(result) is lms


# ── body_visible() ────────────────────────────────────────────────────────────

def test_body_visible_returns_false_on_none():
    detector = PoseDetector(model_path="fake.task")
    assert detector.body_visible(None) is False


def test_body_visible_true_when_all_key_above_threshold():
    detector = PoseDetector(model_path="fake.task")
    lms = _make_33_landmarks(default_visibility=1.0)
    assert detector.body_visible(lms) is True


def test_body_visible_false_when_key_landmark_below_threshold():
    detector = PoseDetector(model_path="fake.task")
    lms = _make_33_landmarks(default_visibility=1.0)
    lms[0].visibility = VISIBILITY_THRESHOLD - 0.01  # nose just below threshold
    assert detector.body_visible(lms) is False


def test_body_visible_true_at_exact_threshold():
    detector = PoseDetector(model_path="fake.task")
    lms = _make_33_landmarks(default_visibility=VISIBILITY_THRESHOLD)
    assert detector.body_visible(lms) is True


def test_body_visible_ignores_non_key_landmarks():
    detector = PoseDetector(model_path="fake.task")
    lms = _make_33_landmarks(default_visibility=1.0)
    # Index 1 (LEFT_EYE_INNER) is not a key landmark
    non_key = set(range(33)) - set(KEY_LANDMARK_INDICES)
    for i in non_key:
        lms[i].visibility = 0.0
    assert detector.body_visible(lms) is True


# ── close() ───────────────────────────────────────────────────────────────────

def test_close_clears_landmarker():
    detector = PoseDetector(model_path="fake.task")
    mock_lm = MagicMock()
    with patch("pose.mp.tasks.vision.PoseLandmarker.create_from_options", return_value=mock_lm):
        detector.open()
    assert detector.is_open() is True
    detector.close()
    mock_lm.close.assert_called_once()
    assert detector._landmarker is None
    assert detector.is_open() is False
