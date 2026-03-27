import numpy as np
import mediapipe as mp

VISIBILITY_THRESHOLD = 0.6

KEY_LANDMARK_INDICES: tuple[int, ...] = (
    0,            # NOSE
    11, 12,       # SHOULDERS
    13, 14,       # ELBOWS
    15, 16,       # WRISTS
    23, 24,       # HIPS
    25, 26,       # KNEES
    27, 28,       # ANKLES
)

KEY_LANDMARK_NAMES: tuple[str, ...] = (
    "Nose",
    "L Shoulder", "R Shoulder",
    "L Elbow",    "R Elbow",
    "L Wrist",    "R Wrist",
    "L Hip",      "R Hip",
    "L Knee",     "R Knee",
    "L Ankle",    "R Ankle",
)


class PoseDetector:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._landmarker = None

    def open(self) -> bool:
        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            return True
        except Exception:
            self._landmarker = None
            return False

    def is_open(self) -> bool:
        return self._landmarker is not None

    def process(self, rgb_frame: np.ndarray, timestamp_ms: int):
        if not self.is_open():
            return None
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            return self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            return None

    def get_landmarks(self, result) -> list | None:
        if result is None or not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def body_visible(self, landmarks: list | None) -> bool:
        if landmarks is None:
            return False
        return all(
            landmarks[i].visibility >= VISIBILITY_THRESHOLD
            for i in KEY_LANDMARK_INDICES
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
