"""Sanity checks for repository config/automixer.yaml (valid YAML, no merge junk)."""

import os

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTOMIXER_PATH = os.path.join(REPO_ROOT, "config", "automixer.yaml")


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
def test_automixer_yaml_parses_and_has_expected_sections():
    """Shipped automixer.yaml must load and contain core sections."""
    with open(AUTOMIXER_PATH, encoding="utf-8") as f:
        raw = f.read()
    assert raw.count("<<<<<<<") == 0, "Unresolved git merge markers in automixer.yaml"
    assert raw.count(">>>>>>>") == 0

    data = yaml.safe_load(raw)
    assert isinstance(data, dict)
    assert "mixer" in data
    assert "audio" in data
    assert data["mixer"].get("type") == "dlive"
    assert data["audio"].get("sample_rate") == 48000
