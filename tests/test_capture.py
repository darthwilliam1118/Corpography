import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capture import Capture


class MockCap:
    def __init__(self, opened: bool = True, frame: np.ndarray | None = None):
        self._opened = opened
        self._frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._opened:
            return True, self._frame.copy()
        return False, None

    def set(self, prop_id, value):
        return True

    def release(self):
        self._opened = False


def _make_cap(opened=True, frame=None):
    mock = MockCap(opened=opened, frame=frame)
    return MagicMock(return_value=mock), mock


# ── construction ──────────────────────────────────────────────────────────────

def test_instantiation_without_camera_hardware():
    cap = Capture()
    assert cap.is_open() is False


# ── open() ────────────────────────────────────────────────────────────────────

def test_open_success():
    ctor, _ = _make_cap(opened=True)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        assert cap.open() is True
        assert cap.is_open() is True


def test_open_failure():
    ctor, _ = _make_cap(opened=False)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        assert cap.open() is False
        assert cap.is_open() is False


# ── get_frame() ───────────────────────────────────────────────────────────────

def test_get_frame_returns_none_when_not_open():
    cap = Capture()
    assert cap.get_frame() is None


def test_get_frame_returns_none_when_read_fails():
    ctor, mock = _make_cap(opened=True)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        cap.open()
        assert cap.is_open()
        # Simulate a mid-run read failure after a successful open
        mock.read = lambda: (False, None)
        assert cap.get_frame() is None


def test_get_frame_returns_rgb_array():
    ctor, _ = _make_cap(opened=True)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        cap.open()
        result = cap.get_frame()
    assert result is not None
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


def test_get_frame_is_horizontally_flipped():
    # Build a frame where the leftmost column is pure blue in BGR: [255, 0, 0]
    # After cv2.flip(frame, 1): blue pixel moves to rightmost column
    # After BGR→RGB: [255, 0, 0] BGR becomes [0, 0, 255] RGB
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, 0] = [255, 0, 0]  # leftmost column = blue in BGR

    ctor, _ = _make_cap(opened=True, frame=frame)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        cap.open()
        result = cap.get_frame()

    assert result is not None
    # Rightmost column should be [0, 0, 255] in RGB (the original blue)
    np.testing.assert_array_equal(result[0, -1], [0, 0, 255])
    # Leftmost column should be black (was the right side of the original black frame)
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0])


# ── release() ─────────────────────────────────────────────────────────────────

def test_release_clears_open_state():
    ctor, _ = _make_cap(opened=True)
    with patch("capture.cv2.VideoCapture", ctor):
        cap = Capture()
        cap.open()
        assert cap.is_open() is True
        cap.release()
        assert cap.is_open() is False
