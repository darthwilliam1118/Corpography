import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.templates import (
    TEMPLATE_LANDMARK_NAMES,
    LandmarkEntry,
    ShapeTemplate,
    default_template,
    load_template,
    save_template,
    template_path,
)


# ---------------------------------------------------------------------------
# default_template
# ---------------------------------------------------------------------------

def test_default_template_has_all_13_landmarks():
    t = default_template("A")
    for name in TEMPLATE_LANDMARK_NAMES:
        assert name in t.landmarks, f"Missing landmark: {name}"


def test_default_template_shape_id():
    t = default_template("T")
    assert t.shape_id == "T"


def test_default_values_in_range():
    t = default_template("X")
    for name, entry in t.landmarks.items():
        assert 0.0 <= entry.x <= 1.0, f"{name}.x out of range"
        assert 0.0 <= entry.y <= 1.0, f"{name}.y out of range"
        assert 0.0 <= entry.weight <= 1.0, f"{name}.weight out of range"


def test_default_difficulty():
    t = default_template("I")
    assert t.difficulty == 2


# ---------------------------------------------------------------------------
# template_path
# ---------------------------------------------------------------------------

def test_template_path_format():
    result = template_path("B", "/some/dir")
    assert result == os.path.join("/some/dir", "B.json")


def test_template_path_uses_shape_id():
    assert template_path("Z", "/base").endswith("Z.json")


# ---------------------------------------------------------------------------
# save_template + load_template roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    original = default_template("A")
    original.landmarks["NOSE"].x = 0.5123
    original.landmarks["LEFT_WRIST"].weight = 0.5
    path = str(tmp_path / "A.json")
    save_template(original, path)
    loaded = load_template(path)
    assert loaded.shape_id == original.shape_id
    assert loaded.display_name == original.display_name
    assert loaded.difficulty == original.difficulty
    assert set(loaded.landmarks.keys()) == set(original.landmarks.keys())
    for name in original.landmarks:
        assert abs(loaded.landmarks[name].x - original.landmarks[name].x) < 1e-3
        assert abs(loaded.landmarks[name].y - original.landmarks[name].y) < 1e-3
        assert abs(loaded.landmarks[name].weight - original.landmarks[name].weight) < 1e-3


def test_floats_rounded_on_save(tmp_path):
    t = default_template("R")
    t.landmarks["NOSE"].x = 1 / 3  # 0.333333...
    path = str(tmp_path / "R.json")
    save_template(t, path)
    with open(path) as f:
        raw = f.read()
    data = json.loads(raw)
    saved_x = str(data["landmarks"]["NOSE"]["x"])
    # Should not have more than 4 decimal places
    if "." in saved_x:
        assert len(saved_x.split(".")[1]) <= 4


def test_save_creates_parent_directories(tmp_path):
    nested = tmp_path / "deep" / "nested" / "dir"
    path = str(nested / "A.json")
    t = default_template("A")
    save_template(t, path)
    assert os.path.exists(path)


def test_save_writes_trailing_newline(tmp_path):
    path = str(tmp_path / "A.json")
    save_template(default_template("A"), path)
    with open(path, "rb") as f:
        content = f.read()
    assert content.endswith(b"\n")


# ---------------------------------------------------------------------------
# load_template error cases
# ---------------------------------------------------------------------------

def test_load_raises_file_not_found():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_template("/nonexistent/path/X.json")


def test_load_raises_on_bad_json(tmp_path):
    import pytest
    path = tmp_path / "bad.json"
    path.write_text("not valid json {{{")
    with pytest.raises(ValueError):
        load_template(str(path))


def test_load_raises_on_out_of_range_x(tmp_path):
    import pytest
    t = default_template("A")
    path = str(tmp_path / "A.json")
    save_template(t, path)
    # Manually corrupt the x value
    with open(path) as f:
        data = json.load(f)
    data["landmarks"]["NOSE"]["x"] = 1.5
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="out of range"):
        load_template(path)


def test_load_raises_on_out_of_range_weight(tmp_path):
    import pytest
    t = default_template("A")
    path = str(tmp_path / "A.json")
    save_template(t, path)
    with open(path) as f:
        data = json.load(f)
    data["landmarks"]["NOSE"]["weight"] = -0.1
    with open(path, "w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError, match="out of range"):
        load_template(path)


def test_load_raises_on_missing_shape_id(tmp_path):
    import pytest
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"display_name": "X", "difficulty": 1, "landmarks": {}}))
    with pytest.raises(ValueError):
        load_template(str(path))


def test_load_raises_on_landmarks_not_dict(tmp_path):
    import pytest
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "shape_id": "A", "display_name": "A", "difficulty": 1,
        "landmarks": ["not", "a", "dict"]
    }))
    with pytest.raises(ValueError):
        load_template(str(path))
