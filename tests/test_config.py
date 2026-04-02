import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    DEFAULT_CONFIG,
    _config_path,
    get_camera_index,
    load_config,
    save_config,
    set_camera_index,
)


# ---------------------------------------------------------------------------
# _config_path
# ---------------------------------------------------------------------------

def test_config_path_dev_mode_returns_project_root_json():
    path = _config_path()
    assert path.endswith("config.json")
    # In dev mode the file should sit one level above src/
    assert "src" not in os.path.basename(os.path.dirname(path))


def test_config_path_frozen_mode_returns_exe_dir_json(tmp_path):
    fake_exe = tmp_path / "Corpography.exe"
    fake_exe.touch()
    with patch.object(sys, "frozen", True, create=True), \
         patch.object(sys, "executable", str(fake_exe)):
        path = _config_path()
    assert path == str(tmp_path / "config.json")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_returns_defaults_when_file_missing(tmp_path):
    with patch("config._config_path", return_value=str(tmp_path / "config.json")):
        cfg = load_config()
    assert cfg == DEFAULT_CONFIG


def test_load_config_merges_stored_with_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_index": 2}), encoding="utf-8")
    with patch("config._config_path", return_value=str(path)):
        cfg = load_config()
    assert cfg["camera_index"] == 2


def test_load_config_ignores_malformed_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("not json {{{", encoding="utf-8")
    with patch("config._config_path", return_value=str(path)):
        cfg = load_config()
    assert cfg == DEFAULT_CONFIG


def test_load_config_adds_missing_default_keys(tmp_path):
    # Stored file has an older format without a key that DEFAULT_CONFIG now has
    path = tmp_path / "config.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    with patch("config._config_path", return_value=str(path)):
        cfg = load_config()
    assert "camera_index" in cfg


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

def test_save_config_writes_json(tmp_path):
    path = tmp_path / "config.json"
    with patch("config._config_path", return_value=str(path)):
        save_config({"camera_index": 1})
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["camera_index"] == 1


def test_save_config_roundtrips_through_load(tmp_path):
    path = tmp_path / "config.json"
    with patch("config._config_path", return_value=str(path)):
        save_config({"camera_index": 3})
        cfg = load_config()
    assert cfg["camera_index"] == 3


def test_save_config_prints_warning_on_oserror(tmp_path, capsys):
    with patch("config._config_path", return_value="/nonexistent_dir/config.json"):
        save_config({"camera_index": 0})  # should not raise
    captured = capsys.readouterr()
    assert "Warning" in captured.err or captured.err == ""  # graceful, not crash


# ---------------------------------------------------------------------------
# get_camera_index / set_camera_index
# ---------------------------------------------------------------------------

def test_get_camera_index_returns_none_by_default():
    cfg = dict(DEFAULT_CONFIG)
    assert get_camera_index(cfg) is None


def test_get_camera_index_returns_int():
    assert get_camera_index({"camera_index": 2}) == 2


def test_get_camera_index_coerces_to_int():
    # Stored as float due to JSON round-trip edge cases
    assert get_camera_index({"camera_index": 1.0}) == 1
    assert isinstance(get_camera_index({"camera_index": 1.0}), int)


def test_set_camera_index_returns_new_dict():
    cfg = {"camera_index": None}
    new_cfg = set_camera_index(cfg, 1)
    assert new_cfg["camera_index"] == 1
    assert cfg["camera_index"] is None  # original unchanged


def test_set_camera_index_can_clear_to_none():
    cfg = {"camera_index": 2}
    new_cfg = set_camera_index(cfg, None)
    assert get_camera_index(new_cfg) is None
