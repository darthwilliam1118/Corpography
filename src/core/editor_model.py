"""
Headless editor state — drag, weight, capture, and save logic.
No pygame, no mediapipe imports at module level.
Fully testable without a display or camera.
"""
from __future__ import annotations

import os

from core.templates import (
    MP_INDEX_TO_NAME,
    ShapeTemplate,
    default_template,
    load_template,
    save_template,
    template_path,
)

# Weight cycle sequence (right-click cycles through these)
_WEIGHT_CYCLE = (1.0, 0.5, 0.0)


class EditorModel:
    """
    Holds all mutable state for one editing session on a single shape.

    Coordinates are always normalized [0,1] in both x and y.
    The caller is responsible for converting between screen pixels and
    normalized space before calling any method here.
    """

    def __init__(self, shape_id: str, templates_dir: str) -> None:
        self.shape_id = shape_id
        self.templates_dir = templates_dir
        self.error_message: str | None = None

        path = template_path(shape_id, templates_dir)
        if os.path.exists(path):
            try:
                self.template = load_template(path)
            except (ValueError, FileNotFoundError) as exc:
                self.error_message = str(exc)
                self.template = default_template(shape_id)
        else:
            self.template = default_template(shape_id)

        self.dragging: str | None = None
        self._drag_offset: tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Drag interaction
    # ------------------------------------------------------------------

    def begin_drag(self, name: str, norm_x: float, norm_y: float) -> None:
        """Start dragging the named landmark. norm_x/y is the click point."""
        if name not in self.template.landmarks:
            return
        entry = self.template.landmarks[name]
        self.dragging = name
        self._drag_offset = (norm_x - entry.x, norm_y - entry.y)

    def update_drag(self, norm_x: float, norm_y: float) -> None:
        """Move the currently-dragged landmark, clamped to [0, 1]."""
        if self.dragging is None:
            return
        raw_x = norm_x - self._drag_offset[0]
        raw_y = norm_y - self._drag_offset[1]
        entry = self.template.landmarks[self.dragging]
        entry.x = max(0.0, min(1.0, raw_x))
        entry.y = max(0.0, min(1.0, raw_y))

    def end_drag(self) -> None:
        """Release the current drag."""
        self.dragging = None
        self._drag_offset = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Weight cycling
    # ------------------------------------------------------------------

    def cycle_weight(self, name: str) -> None:
        """Cycle the named landmark's weight through 1.0 → 0.5 → 0.0 → 1.0."""
        if name not in self.template.landmarks:
            return
        entry = self.template.landmarks[name]
        current = entry.weight
        # Find closest step and advance
        closest = min(_WEIGHT_CYCLE, key=lambda w: abs(w - current))
        idx = _WEIGHT_CYCLE.index(closest)
        entry.weight = _WEIGHT_CYCLE[(idx + 1) % len(_WEIGHT_CYCLE)]

    # ------------------------------------------------------------------
    # Pose capture
    # ------------------------------------------------------------------

    def apply_landmarks_from_mediapipe(self, landmarks: list) -> None:
        """
        Overwrite joint positions from a MediaPipe 33-landmark result list.
        Only updates landmarks that appear in MP_INDEX_TO_NAME.
        Preserves existing weights.
        """
        for idx, name in MP_INDEX_TO_NAME.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            if name in self.template.landmarks:
                self.template.landmarks[name].x = max(0.0, min(1.0, float(lm.x)))
                self.template.landmarks[name].y = max(0.0, min(1.0, float(lm.y)))

    def apply_landmarks_with_visibility(
        self, landmarks: list, visibility_threshold: float
    ) -> None:
        """
        Update template from MediaPipe landmarks, respecting per-landmark visibility.

        - Visible (visibility >= threshold): update position, set weight = 1.0
        - Not visible (visibility < threshold): leave position unchanged, set weight = 0.0
        """
        for idx, name in MP_INDEX_TO_NAME.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            if name not in self.template.landmarks:
                continue
            entry = self.template.landmarks[name]
            if lm.visibility >= visibility_threshold:
                entry.x = max(0.0, min(1.0, float(lm.x)))
                entry.y = max(0.0, min(1.0, float(lm.y)))
                entry.weight = 1.0
            else:
                entry.weight = 0.0  # position unchanged

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def hit_test(self, norm_x: float, norm_y: float, radius: float = 0.025) -> str | None:
        """
        Return the name of the landmark nearest to (norm_x, norm_y) within
        radius (in normalized units), or None if no joint is close enough.
        Uses canvas-space aspect ratio 940×960 ≈ square, so Euclidean distance
        in normalized space is a reasonable approximation.
        """
        best_name: str | None = None
        best_dist = radius
        for name, entry in self.template.landmarks.items():
            dx = norm_x - entry.x
            dy = norm_y - entry.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write the current template to disk."""
        path = template_path(self.shape_id, self.templates_dir)
        save_template(self.template, path)
        self.error_message = None
