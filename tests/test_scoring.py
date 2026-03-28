import math
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.scoring import (
    DEFAULT_D_MAX,
    extract_player_points,
    normalize_points,
    score_pose,
    score_pose_detail,
    score_pose_from_pts,
)
from core.templates import (
    MP_INDEX_TO_NAME,
    TEMPLATE_LANDMARK_NAMES,
    LandmarkEntry,
    ShapeTemplate,
    default_template,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmarks_from_dict(positions: dict[str, tuple[float, float]]) -> list:
    """
    Build a 33-element mock landmark list where the named positions are set
    according to MP_INDEX_TO_NAME and positions dict.
    Remaining slots are (0.5, 0.5).
    """
    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9)] * 33
    name_to_idx = {v: k for k, v in MP_INDEX_TO_NAME.items()}
    lms = list(lms)
    for name, (x, y) in positions.items():
        idx = name_to_idx[name]
        lms[idx] = SimpleNamespace(x=x, y=y, visibility=0.9)
    return lms


def _template_from_positions(
    positions: dict[str, tuple[float, float]],
    weight: float = 1.0,
) -> ShapeTemplate:
    """Build a ShapeTemplate where all given positions have the specified weight."""
    t = default_template("TEST")
    for name, (x, y) in positions.items():
        if name in t.landmarks:
            t.landmarks[name].x = x
            t.landmarks[name].y = y
            t.landmarks[name].weight = weight
    # Set unlisted landmarks to weight 0 so only given positions count
    for name in list(t.landmarks):
        if name not in positions:
            t.landmarks[name].weight = 0.0
    return t


# ---------------------------------------------------------------------------
# extract_player_points
# ---------------------------------------------------------------------------

def test_extract_player_points_returns_13_keys():
    lms = _make_landmarks_from_dict({})  # all at (0.5, 0.5)
    result = extract_player_points(lms)
    assert set(result.keys()) == set(TEMPLATE_LANDMARK_NAMES)


def test_extract_player_points_values():
    lms = _make_landmarks_from_dict({"NOSE": (0.3, 0.1)})
    result = extract_player_points(lms)
    assert abs(result["NOSE"][0] - 0.3) < 1e-6
    assert abs(result["NOSE"][1] - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# normalize_points
# ---------------------------------------------------------------------------

def test_normalize_maps_to_unit_square():
    pts = {"A": (0.2, 0.1), "B": (0.8, 0.9), "C": (0.5, 0.5)}
    norm = normalize_points(pts)
    xs = [x for x, _ in norm.values()]
    ys = [y for _, y in norm.values()]
    assert abs(min(xs)) < 1e-6
    assert abs(max(xs) - 1.0) < 1e-6
    assert abs(min(ys)) < 1e-6
    assert abs(max(ys) - 1.0) < 1e-6


def test_normalize_handles_single_point_no_crash():
    pts = {"A": (0.4, 0.6)}
    norm = normalize_points(pts)  # must not raise
    assert "A" in norm


def test_normalize_empty_returns_empty():
    assert normalize_points({}) == {}


# ---------------------------------------------------------------------------
# score_pose — boundary cases
# ---------------------------------------------------------------------------

def test_none_landmarks_scores_zero():
    t = default_template("A")
    assert score_pose(None, t) == 0.0


def test_all_zero_weights_scores_zero():
    t = default_template("A")
    for entry in t.landmarks.values():
        entry.weight = 0.0
    lms = _make_landmarks_from_dict({})
    assert score_pose(lms, t) == 0.0


# ---------------------------------------------------------------------------
# score_pose — correctness
# ---------------------------------------------------------------------------

def test_perfect_match_scores_100():
    # Player landmarks exactly match template positions → score must be 100
    positions = {
        "NOSE":           (0.50, 0.10),
        "LEFT_SHOULDER":  (0.35, 0.30),
        "RIGHT_SHOULDER": (0.65, 0.30),
        "LEFT_HIP":       (0.40, 0.60),
        "RIGHT_HIP":      (0.60, 0.60),
    }
    t = _template_from_positions(positions)
    lms = _make_landmarks_from_dict(positions)
    result = score_pose(lms, t)
    assert abs(result - 100.0) < 0.1


def test_inverted_shape_scores_zero():
    # Template: A top-left, B top-right, C bottom-left, D bottom-right (square corners)
    # Player:   corners swapped — completely wrong shape → distance > D_MAX in normalised space
    positions_template = {
        "LEFT_SHOULDER":  (0.0, 0.0),   # top-left
        "RIGHT_SHOULDER": (1.0, 0.0),   # top-right
        "LEFT_HIP":       (0.0, 1.0),   # bottom-left
        "RIGHT_HIP":      (1.0, 1.0),   # bottom-right
    }
    positions_player = {
        "LEFT_SHOULDER":  (1.0, 1.0),   # swapped to bottom-right
        "RIGHT_SHOULDER": (0.0, 1.0),   # swapped to bottom-left
        "LEFT_HIP":       (1.0, 0.0),   # swapped to top-right
        "RIGHT_HIP":      (0.0, 0.0),   # swapped to top-left
    }
    t = _template_from_positions(positions_template)
    lms = _make_landmarks_from_dict(positions_player)
    # After normalisation both sets map to the same bounding box [0,1]x[0,1]
    # but each individual joint is at the opposite corner → dist = sqrt(2) ≈ 1.41 > D_MAX=0.5
    result = score_pose(lms, t, d_max=DEFAULT_D_MAX)
    assert result == 0.0


def test_zero_weight_joint_ignored_in_total():
    # Shoulders are active (weight=1.0) and player matches them perfectly.
    # NOSE has weight=0 and player NOSE is far off — must not affect score.
    active_positions = {
        "LEFT_SHOULDER":  (0.3, 0.3),
        "RIGHT_SHOULDER": (0.7, 0.3),
    }
    t = _template_from_positions(active_positions, weight=1.0)
    # NOSE still exists in the template at default position with weight 0
    # (already set to 0 by _template_from_positions for unlisted keys)
    player_pos = dict(active_positions)
    player_pos["NOSE"] = (0.99, 0.99)  # far off but weight=0 → ignored
    lms = _make_landmarks_from_dict(player_pos)
    result = score_pose(lms, t)
    assert result > 95.0  # only shoulders scored, player matches them exactly


def test_partial_match_is_between_0_and_100():
    positions = {
        "LEFT_SHOULDER":  (0.2, 0.3),
        "RIGHT_SHOULDER": (0.8, 0.3),
        "LEFT_HIP":       (0.3, 0.7),
        "RIGHT_HIP":      (0.7, 0.7),
    }
    t = _template_from_positions(positions)
    # Player matches left side perfectly, right side is way off
    player_pos = dict(positions)
    player_pos["RIGHT_SHOULDER"] = (0.2, 0.9)
    player_pos["RIGHT_HIP"] = (0.2, 0.9)
    lms = _make_landmarks_from_dict(player_pos)
    result = score_pose(lms, t)
    assert 0.0 < result < 100.0


def test_d_max_param_controls_sensitivity():
    positions = {
        "LEFT_SHOULDER":  (0.3, 0.3),
        "RIGHT_SHOULDER": (0.7, 0.3),
        "LEFT_HIP":       (0.35, 0.7),
        "RIGHT_HIP":      (0.65, 0.7),
    }
    t = _template_from_positions(positions)
    # Slightly imperfect player
    player_pos = {k: (x + 0.1, y + 0.1) for k, (x, y) in positions.items()}
    lms = _make_landmarks_from_dict(player_pos)
    score_strict = score_pose(lms, t, d_max=0.15)   # tight tolerance
    score_lenient = score_pose(lms, t, d_max=1.0)   # lenient
    assert score_strict < score_lenient


# ---------------------------------------------------------------------------
# score_pose_from_pts and score_pose_detail
# ---------------------------------------------------------------------------

def test_score_pose_from_pts_perfect_match():
    """player_pts == template positions → total == 100, all per_joint == 100."""
    positions = {
        "NOSE":           (0.50, 0.10),
        "LEFT_SHOULDER":  (0.35, 0.30),
        "RIGHT_SHOULDER": (0.65, 0.30),
        "LEFT_HIP":       (0.40, 0.60),
        "RIGHT_HIP":      (0.60, 0.60),
    }
    t = _template_from_positions(positions)
    total, per_joint = score_pose_from_pts(positions, t)
    assert abs(total - 100.0) < 0.1
    for name, s in per_joint.items():
        assert abs(s - 100.0) < 0.1, f"{name} expected 100, got {s}"


def test_score_pose_from_pts_per_joint_keys_match_active():
    """per_joint dict contains only the weight>0 landmark names."""
    positions = {
        "LEFT_SHOULDER":  (0.3, 0.3),
        "RIGHT_SHOULDER": (0.7, 0.3),
    }
    t = _template_from_positions(positions, weight=1.0)
    _, per_joint = score_pose_from_pts(positions, t)
    active_names = {n for n, e in t.landmarks.items() if e.weight > 0}
    assert set(per_joint.keys()) == active_names


def test_score_pose_detail_consistent_with_score_pose():
    """score_pose_detail total matches score_pose for the same MediaPipe input."""
    positions = {
        "NOSE":           (0.50, 0.10),
        "LEFT_SHOULDER":  (0.35, 0.30),
        "RIGHT_SHOULDER": (0.65, 0.30),
        "LEFT_HIP":       (0.40, 0.60),
        "RIGHT_HIP":      (0.60, 0.60),
    }
    t = _template_from_positions(positions)
    lms = _make_landmarks_from_dict(positions)
    detail_total, _ = score_pose_detail(lms, t)
    simple_total = score_pose(lms, t)
    assert abs(detail_total - simple_total) < 1e-9


def test_bounding_box_normalization_size_invariance():
    """Same shape at different scales/offsets should score the same."""
    base = {
        "LEFT_SHOULDER":  (0.3, 0.3),
        "RIGHT_SHOULDER": (0.7, 0.3),
        "LEFT_HIP":       (0.35, 0.7),
        "RIGHT_HIP":      (0.65, 0.7),
    }
    t = _template_from_positions(base)

    # Player 1: exact same shape as template
    lms1 = _make_landmarks_from_dict(base)
    score1 = score_pose(lms1, t)

    # Player 2: same shape but scaled to half size + offset
    scaled = {k: (x * 0.5 + 0.1, y * 0.5 + 0.1) for k, (x, y) in base.items()}
    lms2 = _make_landmarks_from_dict(scaled)
    score2 = score_pose(lms2, t)

    assert abs(score1 - score2) < 1.0  # within 1 point
