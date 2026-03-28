"""
Pose scoring — pure functions, no pygame, no mediapipe imports.

Algorithm:
  1. Extract the 13 key player landmark positions from a MediaPipe result list.
  2. Bounding-box normalise both the player and template point sets independently,
     so that body size / camera distance don't affect the score.
  3. Per landmark: score = clamp(1 - distance / D_MAX, 0, 1).
  4. Weighted average → multiply by 100 → [0, 100].
"""
from __future__ import annotations

import math

from core.templates import MP_INDEX_TO_NAME, ShapeTemplate

DEFAULT_D_MAX: float = 0.5
_EPS: float = 1e-6


def extract_player_points(landmarks: list) -> dict[str, tuple[float, float]]:
    """
    Convert a MediaPipe 33-landmark result list to {name: (x, y)}
    for our 13 key points, using MP_INDEX_TO_NAME.
    """
    return {
        name: (float(landmarks[idx].x), float(landmarks[idx].y))
        for idx, name in MP_INDEX_TO_NAME.items()
        if idx < len(landmarks)
    }


def normalize_points(
    points: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    """
    Bounding-box normalise a named point set to [0,1]×[0,1].
    If the set has fewer than 2 points (or zero span), returns it unchanged.
    """
    if len(points) < 2:
        return {name: (0.0, 0.0) for name in points}

    xs = [x for x, _ in points.values()]
    ys = [y for _, y in points.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = max(x_max - x_min, _EPS)
    span_y = max(y_max - y_min, _EPS)

    return {
        name: ((x - x_min) / span_x, (y - y_min) / span_y)
        for name, (x, y) in points.items()
    }


def score_pose_from_pts(
    player_pts: dict[str, tuple[float, float]] | None,
    template: ShapeTemplate,
    d_max: float = DEFAULT_D_MAX,
) -> tuple[float, dict[str, float]]:
    """
    Core scoring using pre-extracted {name: (x, y)} player points.

    Returns (total_score 0–100, {name: joint_score 0–100}).
    The per-joint dict contains only active (weight > 0) joints.

    Returns (0.0, {}) if player_pts is None or total template weight is zero.
    Normalisation uses only the active landmarks so that:
    - A perfect shape match always scores 100 regardless of body size.
    - Zero-weight landmarks are fully excluded from both normalisation and scoring.
    """
    if player_pts is None:
        return 0.0, {}

    active = {
        name: entry
        for name, entry in template.landmarks.items()
        if entry.weight > 0.0
    }
    if not active:
        return 0.0, {}

    player_raw: dict[str, tuple[float, float]] = {
        name: player_pts[name] for name in active if name in player_pts
    }
    template_raw: dict[str, tuple[float, float]] = {
        name: (entry.x, entry.y) for name, entry in active.items()
    }

    if not player_raw:
        return 0.0, {}

    player_norm   = normalize_points(player_raw)
    template_norm = normalize_points(template_raw)

    weighted_sum = 0.0
    total_weight = 0.0
    per_joint: dict[str, float] = {}

    for name, entry in active.items():
        if name not in player_norm or name not in template_norm:
            continue
        px, py = player_norm[name]
        tx, ty = template_norm[name]
        dist = math.sqrt((px - tx) ** 2 + (py - ty) ** 2)
        s = max(0.0, 1.0 - dist / max(d_max, _EPS))
        per_joint[name] = s * 100.0
        weighted_sum += entry.weight * s
        total_weight += entry.weight

    if total_weight <= 0.0:
        return 0.0, per_joint

    return 100.0 * weighted_sum / total_weight, per_joint


def score_pose_detail(
    landmarks: list | None,
    template: ShapeTemplate,
    d_max: float = DEFAULT_D_MAX,
) -> tuple[float, dict[str, float]]:
    """
    Like score_pose but also returns the per-joint breakdown.

    Returns (total_score 0–100, {name: joint_score 0–100}).
    """
    pts = extract_player_points(landmarks) if landmarks is not None else None
    return score_pose_from_pts(pts, template, d_max)


def score_pose(
    landmarks: list | None,
    template: ShapeTemplate,
    d_max: float = DEFAULT_D_MAX,
) -> float:
    """
    Compare a player's MediaPipe pose against a ShapeTemplate.

    Returns a float in [0.0, 100.0].
    Returns 0.0 if landmarks is None, or if total template weight is zero.
    """
    pts = extract_player_points(landmarks) if landmarks is not None else None
    total, _ = score_pose_from_pts(pts, template, d_max)
    return total
