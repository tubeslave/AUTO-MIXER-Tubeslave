"""Tests for config_manager module."""
import pytest
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from config_manager import ConfigManager


class TestConfigManager:
    """Tests for the ConfigManager class."""

    def test_defaults_loaded(self):
        """ConfigManager loads all default sections without a config file."""
        cm = ConfigManager()
        assert cm.get('mixer.ip') == '192.168.1.1'
        assert cm.get('mixer.port') == 2222
        assert cm.get('audio.sample_rate') == 48000
        assert cm.get('websocket.port') == 8765
        assert cm.get('agent.mode') == 'suggest'
        assert cm.get('safety.feedback_detection') is True

    def test_get_with_missing_path_returns_default(self):
        """get() returns the provided default when the path does not exist."""
        cm = ConfigManager()
        assert cm.get('nonexistent.key') is None
        assert cm.get('nonexistent.key', 'fallback') == 'fallback'
        assert cm.get('mixer.nonexistent', 42) == 42

    def test_set_and_get_value(self):
        """set() creates or updates a config value retrievable by get()."""
        cm = ConfigManager()
        cm.set('mixer.ip', '10.0.0.5')
        assert cm.get('mixer.ip') == '10.0.0.5'

        # Set a nested path that doesn't exist yet
        cm.set('custom.section.key', 'test_value')
        assert cm.get('custom.section.key') == 'test_value'

    def test_load_yaml_config(self):
        """ConfigManager loads and merges values from a YAML config file."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        config_data = {
            'mixer': {'ip': '10.10.10.10', 'port': 3333},
            'logging': {'level': 'DEBUG'},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            cm = ConfigManager(config_path=yaml_path)
            # Overridden values
            assert cm.get('mixer.ip') == '10.10.10.10'
            assert cm.get('mixer.port') == 3333
            assert cm.get('logging.level') == 'DEBUG'
            # Defaults still present for non-overridden values
            assert cm.get('audio.sample_rate') == 48000
        finally:
            os.unlink(yaml_path)

    def test_get_section_returns_deep_copy(self):
        """get_section returns a deep copy that doesn't mutate the config."""
        cm = ConfigManager()
        mixer = cm.get_section('mixer')
        assert mixer['ip'] == '192.168.1.1'
        mixer['ip'] = 'CHANGED'
        # Original should be unchanged
        assert cm.get('mixer.ip') == '192.168.1.1'

    def test_on_change_callback(self):
        """on_change callbacks are called when set() is invoked."""
        cm = ConfigManager()
        callback_calls = []
        cm.on_change(lambda config: callback_calls.append(config.get('mixer', {}).get('ip')))

        cm.set('mixer.ip', '172.16.0.1')
        assert len(callback_calls) == 1
        assert callback_calls[0] == '172.16.0.1'

    def test_to_dict_returns_full_config(self):
        """to_dict returns the full configuration as a dict."""
        cm = ConfigManager()
        d = cm.to_dict()
        assert isinstance(d, dict)
        assert 'mixer' in d
        assert 'audio' in d
        assert 'agent' in d
        assert 'safety' in d
        # Verify it's a deep copy
        d['mixer']['ip'] = 'MUTATED'
        assert cm.get('mixer.ip') != 'MUTATED'

    def test_env_var_override(self):
        """Environment variables with AUTOMIXER_ prefix override config values."""
        original = os.environ.get('AUTOMIXER_MIXER_IP')
        try:
            os.environ['AUTOMIXER_MIXER_IP'] = '192.168.99.99'
            cm = ConfigManager()
            assert cm.get('mixer.ip') == '192.168.99.99'
        finally:
            if original is None:
                os.environ.pop('AUTOMIXER_MIXER_IP', None)
            else:
                os.environ['AUTOMIXER_MIXER_IP'] = original

    def test_model_fallbacks_env_override(self, monkeypatch):
        """AUTOMIXER_MODEL_FALLBACKS pins the active LLM fallback chain."""
        monkeypatch.setenv(
            'AUTOMIXER_MODEL_FALLBACKS',
            'ollama:kimi-k2.5:cloud, ollama:qwen3:0.6b',
        )

        cm = ConfigManager()

        assert cm.get('ai.model_fallbacks') == [
            'ollama:kimi-k2.5:cloud',
            'ollama:qwen3:0.6b',
        ]

    def test_default_model_fallbacks_include_kimi_secondary(self):
        """Default config wires Kimi as the second backend in the LLM chain."""
        cm = ConfigManager()

        assert cm.get('ai.model_fallbacks')[:3] == [
            'openai:gpt-5.4',
            'kimi_cli:default',
            'openai:gpt-4o-mini',
        ]
