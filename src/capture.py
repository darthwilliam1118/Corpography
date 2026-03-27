import cv2
import numpy as np


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
