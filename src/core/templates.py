"""
Shared pose template data model — load, save, and default templates.

Pure stdlib only (json, os, dataclasses). No pygame, no mediapipe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Landmark name constants
# ---------------------------------------------------------------------------

TEMPLATE_LANDMARK_NAMES: tuple[str, ...] = (
    "NOSE",
    "LEFT_SHOULDER",  "RIGHT_SHOULDER",
    "LEFT_ELBOW",     "RIGHT_ELBOW",
    "LEFT_WRIST",     "RIGHT_WRIST",
    "LEFT_HIP",       "RIGHT_HIP",
    "LEFT_KNEE",      "RIGHT_KNEE",
    "LEFT_ANKLE",     "RIGHT_ANKLE",
)

# Maps MediaPipe landmark list index → template name string.
# Single source of truth — must match KEY_LANDMARK_INDICES in pose.py.
MP_INDEX_TO_NAME: dict[int, str] = {
    0:  "NOSE",
    11: "LEFT_SHOULDER",  12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",     14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",     16: "RIGHT_WRIST",
    23: "LEFT_HIP",       24: "RIGHT_HIP",
    25: "LEFT_KNEE",      26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",     28: "RIGHT_ANKLE",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LandmarkEntry:
    x: float      # normalized 0.0–1.0, left=0 right=1
    y: float      # normalized 0.0–1.0, top=0 bottom=1
    weight: float # 0.0 = ignore entirely, 1.0 = full scoring weight


@dataclass
class ShapeTemplate:
    shape_id: str
    display_name: str
    difficulty: int                          # 1–3
    landmarks: dict[str, LandmarkEntry] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default neutral pose
# ---------------------------------------------------------------------------

_NEUTRAL_POSITIONS: dict[str, tuple[float, float]] = {
    "NOSE":            (0.50, 0.12),
    "LEFT_SHOULDER":   (0.35, 0.35),
    "RIGHT_SHOULDER":  (0.65, 0.35),
    "LEFT_ELBOW":      (0.22, 0.52),
    "RIGHT_ELBOW":     (0.78, 0.52),
    "LEFT_WRIST":      (0.18, 0.70),
    "RIGHT_WRIST":     (0.82, 0.70),
    "LEFT_HIP":        (0.40, 0.65),
    "RIGHT_HIP":       (0.60, 0.65),
    "LEFT_KNEE":       (0.40, 0.82),
    "RIGHT_KNEE":      (0.60, 0.82),
    "LEFT_ANKLE":      (0.40, 0.95),
    "RIGHT_ANKLE":     (0.60, 0.95),
}


def default_template(shape_id: str) -> ShapeTemplate:
    """Return a ShapeTemplate with all joints at neutral standing positions, weight 1.0."""
    landmarks = {
        name: LandmarkEntry(x=x, y=y, weight=1.0)
        for name, (x, y) in _NEUTRAL_POSITIONS.items()
    }
    return ShapeTemplate(
        shape_id=shape_id,
        display_name=f"Letter {shape_id}",
        difficulty=2,
        landmarks=landmarks,
    )


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def template_path(shape_id: str, base_dir: str) -> str:
    """Resolve canonical path: <base_dir>/<shape_id>.json"""
    return os.path.join(base_dir, f"{shape_id}.json")


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_template(path: str) -> ShapeTemplate:
    """
    Load a ShapeTemplate from a JSON file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the JSON is malformed or contains out-of-range values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in template {path}: {exc}") from exc

    try:
        shape_id = str(data["shape_id"])
        display_name = str(data["display_name"])
        difficulty = int(data["difficulty"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing required field in {path}: {exc}") from exc

    landmarks: dict[str, LandmarkEntry] = {}
    raw_landmarks = data.get("landmarks", {})
    if not isinstance(raw_landmarks, dict):
        raise ValueError(f"'landmarks' must be a dict in {path}")

    for name, entry in raw_landmarks.items():
        try:
            x = float(entry["x"])
            y = float(entry["y"])
            w = float(entry["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Bad landmark entry '{name}' in {path}: {exc}") from exc

        if not (0.0 <= x <= 1.0):
            raise ValueError(f"Landmark '{name}' x={x} out of range [0,1] in {path}")
        if not (0.0 <= y <= 1.0):
            raise ValueError(f"Landmark '{name}' y={y} out of range [0,1] in {path}")
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"Landmark '{name}' weight={w} out of range [0,1] in {path}")

        landmarks[name] = LandmarkEntry(x=x, y=y, weight=w)

    return ShapeTemplate(
        shape_id=shape_id,
        display_name=display_name,
        difficulty=difficulty,
        landmarks=landmarks,
    )


def save_template(template: ShapeTemplate, path: str) -> None:
    """
    Write a ShapeTemplate to a JSON file.
    Creates parent directories if needed.
    Floats are rounded to 4 decimal places for clean diffs.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    landmarks_dict = {
        name: {
            "x":      round(entry.x, 4),
            "y":      round(entry.y, 4),
            "weight": round(entry.weight, 4),
        }
        for name, entry in template.landmarks.items()
    }

    payload = {
        "shape_id":     template.shape_id,
        "display_name": template.display_name,
        "difficulty":   template.difficulty,
        "landmarks":    landmarks_dict,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
