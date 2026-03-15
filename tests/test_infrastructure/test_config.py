"""
Tests for backend/config_manager.py — additional config integration tests.

See also test_config_manager.py for comprehensive ConfigManager unit tests.
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
    return ConfigManager()


@pytest.fixture
def json_config_file():
    """Temporary JSON config file."""
    data = {
        "mixer": {
            "ip": "192.168.1.102",
            "port": 2223,
            "channels": 40
        },
        "audio": {
            "sample_rate": 48000,
            "buffer_size": 1024
        },
        "safety": {
            "max_gain_db": 10.0,
            "feedback_detection": True
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def yaml_config_file():
    """Temporary YAML config file."""
    data = {
        "mixer": {"ip": "10.0.0.1", "port": 2223},
        "lufs": {"target": -18.0, "tolerance": 1.5},
        "genres": {"rock": {"bass_boost": 3.0}},
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        if HAS_YAML:
            yaml.dump(data, f)
        else:
            json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# JSON config loading
# ---------------------------------------------------------------------------

class TestJsonConfig:
    def test_load_json(self, config_mgr, json_config_file):
        config = config_mgr.load(json_config_file)
        assert config["mixer"]["ip"] == "192.168.1.102"

    def test_dot_notation_get(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        assert config_mgr.get("mixer.ip") == "192.168.1.102"
        assert config_mgr.get("mixer.port") == 2223

    def test_nested_dot_notation(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        assert config_mgr.get("audio.sample_rate") == 48000

    def test_missing_key_returns_default(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        assert config_mgr.get("nonexistent.key", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
class TestYamlConfig:
    def test_load_yaml(self, config_mgr, yaml_config_file):
        config = config_mgr.load(yaml_config_file)
        assert config["mixer"]["ip"] == "10.0.0.1"

    def test_yaml_dot_notation(self, config_mgr, yaml_config_file):
        config_mgr.load(yaml_config_file)
        assert config_mgr.get("lufs.target") == -18.0


# ---------------------------------------------------------------------------
# Config set and callbacks
# ---------------------------------------------------------------------------

class TestConfigSet:
    def test_set_value(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        config_mgr.set("mixer.ip", "10.0.0.5")
        assert config_mgr.get("mixer.ip") == "10.0.0.5"

    def test_set_creates_intermediate(self, config_mgr):
        config_mgr.set("new.nested.key", 42)
        assert config_mgr.get("new.nested.key") == 42

    def test_change_callback_fires(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        changes = []
        config_mgr.on_change(lambda k, old, new: changes.append((k, old, new)))
        config_mgr.set("mixer.ip", "10.0.0.99")
        assert len(changes) == 1
        assert changes[0][0] == "mixer.ip"
        assert changes[0][2] == "10.0.0.99"

    def test_no_callback_on_same_value(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        changes = []
        config_mgr.on_change(lambda k, old, new: changes.append(1))
        config_mgr.set("mixer.port", 2223)  # Same value
        assert len(changes) == 0


# ---------------------------------------------------------------------------
# Config save
# ---------------------------------------------------------------------------

class TestConfigSave:
    def test_save_json(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        config_mgr.set("mixer.ip", "10.10.10.10")
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            out_path = f.name
        try:
            config_mgr.save(out_path)
            with open(out_path) as f:
                saved = json.load(f)
            assert saved["mixer"]["ip"] == "10.10.10.10"
        finally:
            os.unlink(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_required_missing(self, config_mgr):
        config_mgr.set("mixer.ip", "10.0.0.1")
        schema = {
            "mixer": {
                "required": True,
                "type": "dict",
                "children": {
                    "ip": {"required": True, "type": "str"},
                    "port": {"required": True, "type": "int"},
                }
            }
        }
        errors = config_mgr.validate(schema)
        assert any("port" in e for e in errors)

    def test_validate_type_mismatch(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        schema = {
            "mixer": {
                "required": True,
                "type": "dict",
                "children": {
                    "ip": {"required": True, "type": "int"},  # Wrong type
                }
            }
        }
        errors = config_mgr.validate(schema)
        assert len(errors) > 0

    def test_validate_passes(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        schema = {
            "mixer": {
                "required": True,
                "type": "dict",
                "children": {
                    "ip": {"required": True, "type": "str"},
                    "port": {"required": True, "type": "int"},
                }
            }
        }
        errors = config_mgr.validate(schema)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------

class TestAsDict:
    def test_returns_copy(self, config_mgr, json_config_file):
        config_mgr.load(json_config_file)
        d1 = config_mgr.as_dict()
        d1["mixer"]["ip"] = "modified"
        d2 = config_mgr.as_dict()
        assert d2["mixer"]["ip"] != "modified"
