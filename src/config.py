"""
Corpography persistent configuration.

Stores user settings in config.json next to the executable (bundled) or at the
project root (dev mode). New keys can be added to DEFAULT_CONFIG without breaking
existing saved files — load_config merges stored data against the defaults.
"""
from __future__ import annotations

import json
import os
import sys

DEFAULT_CONFIG: dict = {
    "camera_index": None,  # int | None; None = not yet chosen
}


def _config_path() -> str:
    """Return absolute path to config.json."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        # src/config.py → project root is one level up
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, "config.json")


def load_config() -> dict:
    """Read config.json and return it merged with DEFAULT_CONFIG.

    If the file does not exist or is malformed, returns a fresh copy of
    DEFAULT_CONFIG so callers never have to handle None.
    """
    cfg = dict(DEFAULT_CONFIG)
    path = _config_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            cfg.update(stored)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return cfg


def save_config(config: dict) -> None:
    """Write config dict to config.json atomically (tmp file → rename).

    Prints a warning on OSError rather than crashing the app.
    """
    path = _config_path()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp_path, path)
    except OSError as exc:
        print(f"[config] Warning: could not save config to {path}: {exc}", file=sys.stderr)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_camera_index(config: dict) -> int | None:
    """Return the saved camera device index, or None if not yet set."""
    value = config.get("camera_index")
    return int(value) if value is not None else None


def set_camera_index(config: dict, index: int | None) -> dict:
    """Return a new config dict with camera_index set. Does not write to disk."""
    updated = dict(config)
    updated["camera_index"] = index
    return updated
