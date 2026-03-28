import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.editor_model import EditorModel
from core.templates import (
    TEMPLATE_LANDMARK_NAMES,
    MP_INDEX_TO_NAME,
    default_template,
    save_template,
    template_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(tmp_path, shape_id="A"):
    return EditorModel(shape_id=shape_id, templates_dir=str(tmp_path))


def _fake_landmarks():
    """Return a 33-element list of SimpleNamespace(x, y, visibility) mocks."""
    lm = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9)] * 33
    # Set specific values for our key indices
    lm[11] = SimpleNamespace(x=0.30, y=0.40, visibility=0.95)  # LEFT_SHOULDER
    lm[12] = SimpleNamespace(x=0.70, y=0.40, visibility=0.95)  # RIGHT_SHOULDER
    lm[0]  = SimpleNamespace(x=0.50, y=0.10, visibility=0.99)  # NOSE
    return lm


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_uses_default_when_no_file(tmp_path):
    model = _make_model(tmp_path, "X")
    assert "NOSE" in model.template.landmarks
    assert model.template.shape_id == "X"


def test_loads_existing_template_when_file_present(tmp_path):
    t = default_template("B")
    t.landmarks["NOSE"].x = 0.1234
    save_template(t, template_path("B", str(tmp_path)))
    model = EditorModel(shape_id="B", templates_dir=str(tmp_path))
    assert abs(model.template.landmarks["NOSE"].x - 0.1234) < 1e-3


def test_no_error_message_on_clean_init(tmp_path):
    model = _make_model(tmp_path)
    assert model.error_message is None


# ---------------------------------------------------------------------------
# Drag
# ---------------------------------------------------------------------------

def test_begin_drag_sets_dragging(tmp_path):
    model = _make_model(tmp_path)
    model.begin_drag("LEFT_WRIST", 0.5, 0.5)
    assert model.dragging == "LEFT_WRIST"


def test_end_drag_clears_dragging(tmp_path):
    model = _make_model(tmp_path)
    model.begin_drag("LEFT_WRIST", 0.5, 0.5)
    model.end_drag()
    assert model.dragging is None


def test_begin_drag_unknown_name_is_noop(tmp_path):
    model = _make_model(tmp_path)
    model.begin_drag("NONEXISTENT_JOINT", 0.5, 0.5)
    assert model.dragging is None


def test_update_drag_moves_joint(tmp_path):
    model = _make_model(tmp_path)
    model.begin_drag("NOSE", 0.5, 0.12)  # click right on the joint
    model.update_drag(0.6, 0.2)
    assert abs(model.template.landmarks["NOSE"].x - 0.6) < 1e-6
    assert abs(model.template.landmarks["NOSE"].y - 0.2) < 1e-6


def test_update_drag_clamps_to_unit_square(tmp_path):
    model = _make_model(tmp_path)
    model.begin_drag("NOSE", 0.5, 0.12)
    model.update_drag(-0.5, 2.0)  # way out of bounds
    entry = model.template.landmarks["NOSE"]
    assert entry.x == 0.0
    assert entry.y == 1.0


def test_update_drag_without_begin_is_noop(tmp_path):
    model = _make_model(tmp_path)
    original_x = model.template.landmarks["NOSE"].x
    model.update_drag(0.9, 0.9)  # no drag started
    assert model.template.landmarks["NOSE"].x == original_x


# ---------------------------------------------------------------------------
# Weight cycling
# ---------------------------------------------------------------------------

def test_cycle_weight_sequence(tmp_path):
    model = _make_model(tmp_path)
    model.template.landmarks["NOSE"].weight = 1.0
    model.cycle_weight("NOSE")
    assert model.template.landmarks["NOSE"].weight == 0.5
    model.cycle_weight("NOSE")
    assert model.template.landmarks["NOSE"].weight == 0.0
    model.cycle_weight("NOSE")
    assert model.template.landmarks["NOSE"].weight == 1.0


def test_cycle_weight_unknown_name_is_noop(tmp_path):
    model = _make_model(tmp_path)
    model.cycle_weight("FAKE_JOINT")  # must not raise


# ---------------------------------------------------------------------------
# apply_landmarks_from_mediapipe
# ---------------------------------------------------------------------------

def test_apply_landmarks_updates_positions(tmp_path):
    model = _make_model(tmp_path)
    lms = _fake_landmarks()
    model.apply_landmarks_from_mediapipe(lms)
    assert abs(model.template.landmarks["LEFT_SHOULDER"].x - 0.30) < 1e-6
    assert abs(model.template.landmarks["LEFT_SHOULDER"].y - 0.40) < 1e-6
    assert abs(model.template.landmarks["NOSE"].x - 0.50) < 1e-6


def test_apply_landmarks_preserves_weights(tmp_path):
    model = _make_model(tmp_path)
    model.template.landmarks["NOSE"].weight = 0.0
    lms = _fake_landmarks()
    model.apply_landmarks_from_mediapipe(lms)
    assert model.template.landmarks["NOSE"].weight == 0.0


def test_apply_landmarks_clamps_values(tmp_path):
    model = _make_model(tmp_path)
    lms = _fake_landmarks()
    lms[0] = SimpleNamespace(x=1.5, y=-0.3, visibility=0.9)  # NOSE out of range
    model.apply_landmarks_from_mediapipe(lms)
    assert model.template.landmarks["NOSE"].x == 1.0
    assert model.template.landmarks["NOSE"].y == 0.0


# ---------------------------------------------------------------------------
# apply_landmarks_with_visibility
# ---------------------------------------------------------------------------

def test_apply_with_visibility_updates_visible_joints(tmp_path):
    model = _make_model(tmp_path)
    lms = _fake_landmarks()
    # All landmarks in _fake_landmarks have visibility=0.9 — above any sane threshold
    old_nose_y = model.template.landmarks["NOSE"].y
    model.apply_landmarks_with_visibility(lms, visibility_threshold=0.6)
    assert abs(model.template.landmarks["NOSE"].x - 0.50) < 1e-6
    assert abs(model.template.landmarks["NOSE"].y - 0.10) < 1e-6
    assert model.template.landmarks["NOSE"].weight == 1.0
    assert model.template.landmarks["LEFT_SHOULDER"].weight == 1.0


def test_apply_with_visibility_zeros_invisible_joints(tmp_path):
    model = _make_model(tmp_path)
    lms = _fake_landmarks()
    lms = list(lms)
    # Make LEFT_ANKLE invisible
    lms[27] = SimpleNamespace(x=0.99, y=0.99, visibility=0.1)
    orig_x = model.template.landmarks["LEFT_ANKLE"].x
    orig_y = model.template.landmarks["LEFT_ANKLE"].y
    model.apply_landmarks_with_visibility(lms, visibility_threshold=0.6)
    # Position must be unchanged
    assert abs(model.template.landmarks["LEFT_ANKLE"].x - orig_x) < 1e-6
    assert abs(model.template.landmarks["LEFT_ANKLE"].y - orig_y) < 1e-6
    # Weight must be zeroed
    assert model.template.landmarks["LEFT_ANKLE"].weight == 0.0


def test_apply_with_visibility_mixed(tmp_path):
    model = _make_model(tmp_path)
    lms = _fake_landmarks()
    lms = list(lms)
    # Make RIGHT_WRIST invisible, keep others visible
    lms[16] = SimpleNamespace(x=0.99, y=0.99, visibility=0.2)
    orig_wrist_x = model.template.landmarks["RIGHT_WRIST"].x
    model.apply_landmarks_with_visibility(lms, visibility_threshold=0.6)
    # RIGHT_WRIST position unchanged, weight zeroed
    assert abs(model.template.landmarks["RIGHT_WRIST"].x - orig_wrist_x) < 1e-6
    assert model.template.landmarks["RIGHT_WRIST"].weight == 0.0
    # NOSE visible → updated, weight 1.0
    assert abs(model.template.landmarks["NOSE"].x - 0.50) < 1e-6
    assert model.template.landmarks["NOSE"].weight == 1.0


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def test_hit_test_finds_joint(tmp_path):
    model = _make_model(tmp_path)
    # NOSE is at (0.50, 0.12) by default — test very close
    result = model.hit_test(0.505, 0.125)
    assert result == "NOSE"


def test_hit_test_returns_none_when_far(tmp_path):
    model = _make_model(tmp_path)
    # No joint is near (0.0, 0.0) in the default layout
    result = model.hit_test(0.0, 0.0, radius=0.005)
    assert result is None


def test_hit_test_radius_respected(tmp_path):
    model = _make_model(tmp_path)
    # NOSE at (0.50, 0.12); test at (0.60, 0.12) — distance 0.10
    result = model.hit_test(0.60, 0.12, radius=0.05)
    assert result is None  # 0.10 > 0.05


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def test_save_writes_json_to_disk(tmp_path):
    model = _make_model(tmp_path, "C")
    model.save()
    path = template_path("C", str(tmp_path))
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data["shape_id"] == "C"


def test_save_and_reload_via_new_model(tmp_path):
    model1 = _make_model(tmp_path, "D")
    model1.template.landmarks["NOSE"].x = 0.42
    model1.template.difficulty = 3
    model1.save()

    model2 = EditorModel(shape_id="D", templates_dir=str(tmp_path))
    assert abs(model2.template.landmarks["NOSE"].x - 0.42) < 1e-3
    assert model2.template.difficulty == 3
