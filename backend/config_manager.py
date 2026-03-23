"""
Configuration manager with YAML support and hot-reload via watchdog.

Note: server.py uses config/default_config.json directly. ConfigManager is used
in tests and may be integrated for hot-reload in future. See config/README.md.
"""
import os
import logging
import copy
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ConfigValue:
    """A configuration value with metadata."""
    value: Any
    default: Any
    description: str = ''
    env_var: Optional[str] = None

class ConfigManager:
    """Manages application configuration with YAML and env vars."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = self._build_defaults()
        self._callbacks: List[Callable] = []
        self._watcher = None

        # Load config
        self._config = copy.deepcopy(self._defaults)
        if config_path and os.path.exists(config_path):
            self._load_yaml(config_path)
        self._apply_env_vars()

        logger.info(f"Config loaded: {len(self._config)} settings")

    def _build_defaults(self) -> Dict[str, Any]:
        """Build default configuration."""
        return {
            'mixer': {
                'type': 'wing',
                'ip': '192.168.1.1',
                'port': 2222,
                'protocol': 'osc',
                'tls': False,
                'midi_base_channel': 0,
                'user_profile': 0,
                'password': '',
                'keepalive_interval': 5.0,
                'command_rate_limit': 10.0,
                'connection_timeout': 10.0,
            },
            'audio': {
                'sample_rate': 48000,
                'block_size': 1024,
                'channels': 40,
                'source': 'dante',
                'device_name': '',
                'device_type': 'default',
                'buffer_seconds': 5.0,
            },
            'websocket': {
                'host': '0.0.0.0',
                'port': 8765,
                'max_clients': 10,
            },
            'agent': {
                'enabled': True,
                'mode': 'suggest',
                'cycle_interval': 0.5,
                'confidence_threshold': 0.6,
                'max_actions_per_cycle': 5,
            },
            'ai': {
                'llm_backend': 'ollama',
                'llm_model': 'llama3',
                'ollama_url': 'http://localhost:11434',
                'perplexity_api_key': '',
                'knowledge_dir': '',
            },
            'safety': {
                'feedback_detection': True,
                'max_gain_db': 10.0,
                'true_peak_limit': -1.0,
                'max_notch_filters': 8,
                'max_notch_depth_db': -12.0,
            },
            'logging': {
                'level': 'INFO',
                'json_output': False,
                'log_file': '',
            },
            'metrics': {
                'enabled': False,
                'prometheus_port': 9090,
            },
            'session': {
                'auto_save': True,
                'save_interval': 300,
                'session_dir': 'sessions',
            },
        }

    def _load_yaml(self, path: str):
        """Load YAML config file."""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            self._deep_merge(self._config, data)
            logger.info(f"Loaded YAML config from {path}")
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML config")
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_vars(self):
        """Apply environment variable overrides (AUTOMIXER_ prefix)."""
        env_map = {
            'AUTOMIXER_MIXER_TYPE': ('mixer', 'type'),
            'AUTOMIXER_MIXER_IP': ('mixer', 'ip'),
            'AUTOMIXER_MIXER_PORT': ('mixer', 'port', int),
            'AUTOMIXER_WS_PORT': ('websocket', 'port', int),
            'AUTOMIXER_LOG_LEVEL': ('logging', 'level'),
            'AUTOMIXER_AGENT_MODE': ('agent', 'mode'),
            'AUTOMIXER_LLM_BACKEND': ('ai', 'llm_backend'),
            'AUTOMIXER_PERPLEXITY_KEY': ('ai', 'perplexity_api_key'),
            'AUTOMIXER_SAMPLE_RATE': ('audio', 'sample_rate', int),
        }
        for env_key, path_info in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                section = path_info[0]
                key = path_info[1]
                converter = path_info[2] if len(path_info) > 2 else str
                try:
                    self._config[section][key] = converter(val)
                except (ValueError, KeyError):
                    pass

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path (e.g., 'mixer.ip')."""
        parts = path.split('.')
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, path: str, value: Any):
        """Set config value by dot-separated path."""
        parts = path.split('.')
        current = self._config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        self._notify_callbacks()

    def get_section(self, section: str) -> Dict:
        """Get an entire config section."""
        return copy.deepcopy(self._config.get(section, {}))

    def on_change(self, callback: Callable):
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all change callbacks."""
        for cb in self._callbacks:
            try:
                cb(self._config)
            except Exception as e:
                logger.error(f"Config callback error: {e}")

    def start_watching(self):
        """Start watching config file for changes (requires watchdog)."""
        if not self.config_path:
            return
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            config_dir = os.path.dirname(os.path.abspath(self.config_path))
            config_name = os.path.basename(self.config_path)
            manager = self

            class Handler(FileSystemEventHandler):
                def on_modified(self, event):
                    if os.path.basename(event.src_path) == config_name:
                        logger.info("Config file changed, reloading...")
                        manager._config = copy.deepcopy(manager._defaults)
                        manager._load_yaml(manager.config_path)
                        manager._apply_env_vars()
                        manager._notify_callbacks()

            observer = Observer()
            observer.schedule(Handler(), config_dir, recursive=False)
            observer.daemon = True
            observer.start()
            self._watcher = observer
            logger.info(f"Watching config file: {self.config_path}")
        except ImportError:
            logger.info("watchdog not installed, config hot-reload disabled")

    def to_dict(self) -> Dict:
        """Get full config as dict."""
        return copy.deepcopy(self._config)

    def save(self, path: Optional[str] = None):
        """Save current config to YAML."""
        save_path = path or self.config_path
        if not save_path:
            return
        try:
            import yaml
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Config saved to {save_path}")
        except ImportError:
            logger.warning("PyYAML not installed, cannot save config")
