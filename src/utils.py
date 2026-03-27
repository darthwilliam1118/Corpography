import os
import sys


def resource_path(relative_path: str) -> str:
    """Resolve a path to a bundled asset, compatible with PyInstaller and dev mode."""
    if hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)
