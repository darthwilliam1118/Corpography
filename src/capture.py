import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras as _lib_enumerate_cameras
from cv2_enumerate_cameras.camera_info import CameraInfo  # type: ignore[import-untyped]


def enumerate_cameras() -> list[CameraInfo]:
    """Return physical cameras (vid and pid both non-zero) via DirectShow.

    Uses CAP_DSHOW on Windows, which initialises USB cameras reliably and
    without the hangs that MSMF can cause on some hardware.

    Virtual cameras (OBS, NDI, etc.) report vid=0 / pid=0 and are excluded.

    Each returned CameraInfo has its .index pre-encoded with the backend
    offset (c.index + c.backend) so the value can be passed directly to
    cv2.VideoCapture / Capture(device_index=...) without further arithmetic.
    """
    seen: set[tuple[int, int]] = set()
    result: list[CameraInfo] = []
    for c in _lib_enumerate_cameras(cv2.CAP_DSHOW):
        if (c.vid or 0) == 0 or (c.pid or 0) == 0:
            continue
        key = (c.vid or 0, c.pid or 0)
        if key not in seen:
            seen.add(key)
            # Encode the backend offset so callers can pass .index straight
            # to cv2.VideoCapture (e.g. raw index 2 + CAP_DSHOW 700 = 702).
            c.index += c.backend
            result.append(c)
    return result


class Capture:
    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._cap = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap
        return True

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_frame(self) -> np.ndarray | None:
        if not self.is_open():
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        frame = cv2.flip(frame, 1)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
