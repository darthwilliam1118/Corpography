import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import resource_path


def test_resource_path_returns_string():
    result = resource_path("assets/images")
    assert isinstance(result, str)


def test_resource_path_is_absolute():
    result = resource_path("assets/images")
    assert os.path.isabs(result)


def test_resource_path_joins_correctly():
    result = resource_path("assets/sounds")
    # Normalize separators for cross-platform comparison
    assert os.path.normpath(result).endswith(os.path.join("assets", "sounds"))
