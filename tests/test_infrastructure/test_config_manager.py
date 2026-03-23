"""
Tests for backend/config_manager.py — ConfigManager with YAML loading,
dot-notation get/set, validation, save, and change callbacks.
"""

import os
import pytest

try:
    from config_manager import ConfigManager
except ImportError:
    pytest.skip("config_manager module not importable", allow_module_level=True)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def yaml_config_file(tmp_path):
    """Create a temporary YAML config file."""
    if not HAS_YAML:
        pytest.skip("PyYAML not installed")
    data = {
        "mixer": {
            "ip": "10.0.0.5",
            "port": 3333,
        },
        "audio": {
            "channels": 32,
            "sample_rate": 44100,
        },
    }
    path = os.path.join(tmp_path, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigManagerLoad:

    def test_defaults_without_file(self):
        """ConfigManager without a file should have defaults."""
        mgr = ConfigManager()
        assert mgr.get("mixer.ip") == "192.168.1.1"
        assert mgr.get("audio.sample_rate") == 48000

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_load_yaml(self, yaml_config_file):
        """Loading a YAML file should override defaults."""
        mgr = ConfigManager(config_path=yaml_config_file)
        assert mgr.get("mixer.ip") == "10.0.0.5"
        assert mgr.get("mixer.port") == 3333
        assert mgr.get("audio.channels") == 32
        # Non-overridden defaults should remain
        assert mgr.get("websocket.port") == 8765

    def test_load_missing_file_uses_defaults(self):
        """Loading a nonexistent file should fall back to defaults."""
        mgr = ConfigManager(config_path="/tmp/nonexistent_config_12345.yaml")
        assert mgr.get("mixer.ip") == "192.168.1.1"


class TestConfigManagerGetSet:

    def test_get_dot_notation(self):
        """Dot-notation get should navigate nested keys."""
        mgr = ConfigManager()
        assert mgr.get("mixer.ip") == "192.168.1.1"
        assert mgr.get("websocket.host") == "0.0.0.0"

    def test_get_default(self):
        """Missing key should return the default."""
        mgr = ConfigManager()
        assert mgr.get("nonexistent.key", "fallback") == "fallback"

    def test_set_dot_notation(self):
        """Dot-notation set should update nested values."""
        mgr = ConfigManager()
        mgr.set("mixer.port", 9999)
        assert mgr.get("mixer.port") == 9999

    def test_set_creates_intermediate_dicts(self):
        """Setting a deeply nested key should create intermediate dicts."""
        mgr = ConfigManager()
        mgr.set("a.b.c", 42)
        assert mgr.get("a.b.c") == 42

    def test_get_section(self):
        """get_section should return a deep copy of a section."""
        mgr = ConfigManager()
        mixer = mgr.get_section("mixer")
        mixer["ip"] = "changed"
        assert mgr.get("mixer.ip") == "192.168.1.1"  # unchanged


class TestConfigManagerSave:

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
    def test_save_yaml(self, tmp_path):
        """Saving should write a valid YAML file."""
        mgr = ConfigManager()
        mgr.set("mixer.port", 5555)
        save_path = os.path.join(tmp_path, "saved.yaml")
        mgr.save(save_path)

        with open(save_path) as f:
            saved = yaml.safe_load(f)
        assert saved["mixer"]["port"] == 5555

    def test_save_no_path_no_error(self):
        """Saving without any path should not raise."""
        mgr = ConfigManager()
        mgr.save()  # should silently return


class TestConfigManagerCallbacks:

    def test_on_change_callback(self):
        """Setting a value should trigger change callbacks."""
        mgr = ConfigManager()
        results = []
        mgr.on_change(lambda cfg: results.append(cfg.get("mixer", {}).get("port")))
        mgr.set("mixer.port", 7777)
        assert 7777 in results

    def test_to_dict(self):
        """to_dict should return a copy of the full config."""
        mgr = ConfigManager()
        d = mgr.to_dict()
        assert isinstance(d, dict)
        assert "mixer" in d
        d["mixer"]["ip"] = "changed"
        assert mgr.get("mixer.ip") == "192.168.1.1"  # unchanged
