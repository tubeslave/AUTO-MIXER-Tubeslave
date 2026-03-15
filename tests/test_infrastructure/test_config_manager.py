"""
Tests for backend/config_manager.py — ConfigManager with YAML/JSON loading,
dot-notation get/set, validation, save, and change callbacks.
"""

import json
import os
import tempfile
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
def config_mgr():
    """Fresh ConfigManager instance."""
    return ConfigManager()


@pytest.fixture
def json_config_file(tmp_dir):
    """Create a temporary JSON config file."""
    data = {
        "mixer": {
            "channels": 40,
            "sample_rate": 48000,
        },
        "network": {
            "host": "localhost",
            "port": 8765,
        },
        "debug": True,
    }
    path = os.path.join(tmp_dir, "config.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def yaml_config_file(tmp_dir):
    """Create a temporary YAML config file (skipped if PyYAML missing)."""
    if not HAS_YAML:
        pytest.skip("PyYAML not installed")
    data = {
        "mixer": {
            "channels": 32,
            "sample_rate": 44100,
        },
        "osc": {
            "ip": "192.168.1.100",
            "port": 2223,
        },
    }
    path = os.path.join(tmp_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfigManagerLoad:

    def test_load_json(self, config_mgr, json_config_file):
        """Loading a JSON file should populate the config."""
        result = config_mgr.load(json_config_file)
        assert result["mixer"]["channels"] == 40
        assert result["debug"] is True

    def test_load_yaml(self, config_mgr, yaml_config_file):
        """Loading a YAML file should populate the config."""
        result = config_mgr.load(yaml_config_file)
        assert result["mixer"]["channels"] == 32
        assert result["osc"]["ip"] == "192.168.1.100"

    def test_load_missing_file_returns_empty(self, config_mgr):
        """Loading a nonexistent file should return an empty dict."""
        result = config_mgr.load("/tmp/nonexistent_config_12345.json")
        assert result == {}


class TestConfigManagerGetSet:

    def test_get_dot_notation(self, config_mgr, json_config_file):
        """Dot-notation get should navigate nested keys."""
        config_mgr.load(json_config_file)
        assert config_mgr.get("mixer.channels") == 40
        assert config_mgr.get("network.host") == "localhost"

    def test_get_default(self, config_mgr, json_config_file):
        """Missing key should return the default."""
        config_mgr.load(json_config_file)
        assert config_mgr.get("nonexistent.key", "fallback") == "fallback"

    def test_set_dot_notation(self, config_mgr, json_config_file):
        """Dot-notation set should update nested values."""
        config_mgr.load(json_config_file)
        config_mgr.set("mixer.channels", 48)
        assert config_mgr.get("mixer.channels") == 48

    def test_set_creates_intermediate_dicts(self, config_mgr):
        """Setting a deeply nested key should create intermediate dicts."""
        config_mgr.set("a.b.c", 42)
        assert config_mgr.get("a.b.c") == 42

    def test_get_returns_deep_copy_for_dicts(self, config_mgr, json_config_file):
        """Returned dicts should be copies, not references."""
        config_mgr.load(json_config_file)
        mixer = config_mgr.get("mixer")
        mixer["channels"] = 999
        assert config_mgr.get("mixer.channels") == 40  # unchanged


class TestConfigManagerSave:

    def test_save_json(self, config_mgr, json_config_file, tmp_dir):
        """Saving should write a valid JSON file."""
        config_mgr.load(json_config_file)
        config_mgr.set("mixer.channels", 64)
        save_path = os.path.join(tmp_dir, "saved.json")
        config_mgr.save(save_path)

        with open(save_path) as f:
            saved = json.load(f)
        assert saved["mixer"]["channels"] == 64

    def test_save_to_original_path(self, config_mgr, json_config_file):
        """Saving without a path should use the original load path."""
        config_mgr.load(json_config_file)
        config_mgr.set("debug", False)
        config_mgr.save()

        with open(json_config_file) as f:
            saved = json.load(f)
        assert saved["debug"] is False

    def test_save_raises_without_path(self, config_mgr):
        """Saving without ever loading or specifying a path should raise."""
        with pytest.raises(ValueError):
            config_mgr.save()


class TestConfigManagerCallbacks:

    def test_on_change_fires(self, config_mgr, json_config_file):
        """Setting a value should fire registered change callbacks."""
        config_mgr.load(json_config_file)
        changes = []
        config_mgr.on_change(lambda k, old, new: changes.append((k, old, new)))
        config_mgr.set("mixer.channels", 24)
        assert len(changes) == 1
        assert changes[0] == ("mixer.channels", 40, 24)

    def test_no_callback_on_same_value(self, config_mgr, json_config_file):
        """Setting the same value should not trigger a callback."""
        config_mgr.load(json_config_file)
        changes = []
        config_mgr.on_change(lambda k, old, new: changes.append((k, old, new)))
        config_mgr.set("mixer.channels", 40)
        assert len(changes) == 0


class TestConfigManagerValidation:

    def test_validate_missing_required_key(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        schema = {
            "required_field": {"required": True, "type": "str"},
        }
        errors = config_mgr.validate(schema)
        assert any("required_field" in e for e in errors)

    def test_validate_wrong_type(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        schema = {
            "debug": {"type": "str"},  # debug is bool, not str
        }
        errors = config_mgr.validate(schema)
        assert any("debug" in e for e in errors)

    def test_validate_passes(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        schema = {
            "debug": {"type": "bool"},
        }
        errors = config_mgr.validate(schema)
        assert len(errors) == 0

    def test_as_dict_returns_copy(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        d = config_mgr.as_dict()
        d["mixer"]["channels"] = 999
        assert config_mgr.get("mixer.channels") == 40
