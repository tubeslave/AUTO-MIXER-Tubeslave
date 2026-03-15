"""
YAML configuration with watchdog hot-reload and validation.

Supports dot-notation access, file watching, and change callbacks.
"""

import json
import logging
import os
import threading
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class ConfigManager:
    """YAML/JSON configuration manager with hot-reload support."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._path: Optional[str] = None
        self._lock = threading.RLock()
        self._callbacks: List[Callable] = []
        self._observer = None
        self._schema: Optional[Dict] = None

    def load(self, path: str) -> Dict[str, Any]:
        """Load configuration from a YAML or JSON file."""
        self._path = path
        with self._lock:
            if not os.path.isfile(path):
                logger.warning(f"Config file not found: {path}")
                return {}

            with open(path, "r") as f:
                if path.endswith((".yaml", ".yml")):
                    if HAS_YAML:
                        self._config = yaml.safe_load(f) or {}
                    else:
                        logger.error("PyYAML not installed")
                        return {}
                else:
                    self._config = json.load(f)

        logger.info(f"Config loaded from {path}")
        return deepcopy(self._config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot notation (e.g., 'mixer.channels')."""
        with self._lock:
            parts = key.split(".")
            current = self._config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return deepcopy(current) if isinstance(current, (dict, list)) else current

    def set(self, key: str, value: Any) -> None:
        """Set a value using dot notation."""
        with self._lock:
            parts = key.split(".")
            current = self._config
            for part in parts[:-1]:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            old_value = current.get(parts[-1])
            current[parts[-1]] = value

        if old_value != value:
            self._notify_change(key, old_value, value)

    def save(self, path: Optional[str] = None) -> None:
        """Save current config to file."""
        save_path = path or self._path
        if not save_path:
            raise ValueError("No path specified for saving config")

        with self._lock:
            with open(save_path, "w") as f:
                if save_path.endswith((".yaml", ".yml")) and HAS_YAML:
                    yaml.dump(self._config, f, default_flow_style=False)
                else:
                    json.dump(self._config, f, indent=2)
        logger.info(f"Config saved to {save_path}")

    def watch(self) -> bool:
        """Start file watcher for hot-reload."""
        if not self._path:
            logger.warning("No config path set, cannot watch")
            return False

        if not HAS_WATCHDOG:
            logger.warning("watchdog not installed, hot-reload disabled")
            return False

        if self._observer:
            return True

        config_dir = os.path.dirname(os.path.abspath(self._path))
        config_name = os.path.basename(self._path)

        manager = self

        class ConfigFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(config_name):
                    logger.info(f"Config file modified: {event.src_path}")
                    try:
                        manager.load(manager._path)
                        manager._notify_change("__reload__", None, None)
                    except Exception as e:
                        logger.error(f"Error reloading config: {e}")

        self._observer = Observer()
        self._observer.schedule(ConfigFileHandler(), config_dir, recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.info(f"Watching config file: {self._path}")
        return True

    def stop_watch(self) -> None:
        """Stop the file watcher."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None

    def on_change(self, callback: Callable) -> None:
        """Register a callback for config changes. Callback receives (key, old, new)."""
        self._callbacks.append(callback)

    def _notify_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify registered callbacks of a config change."""
        for cb in self._callbacks:
            try:
                cb(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Config change callback error: {e}")

    def validate(self, schema: Optional[Dict] = None) -> List[str]:
        """Validate config against expected schema. Returns list of errors."""
        check_schema = schema or self._schema
        if not check_schema:
            return []

        errors = []
        with self._lock:
            self._validate_node(self._config, check_schema, "", errors)
        return errors

    def set_schema(self, schema: Dict) -> None:
        """Set the expected config schema for validation."""
        self._schema = schema

    def _validate_node(self, config: Any, schema: Dict, path: str,
                       errors: List[str]) -> None:
        """Recursively validate config against schema."""
        if not isinstance(schema, dict):
            return

        for key, spec in schema.items():
            full_path = f"{path}.{key}" if path else key
            if key not in config if isinstance(config, dict) else True:
                if spec.get("required", False):
                    errors.append(f"Missing required key: {full_path}")
                continue

            value = config[key] if isinstance(config, dict) else None
            expected_type = spec.get("type")
            if expected_type and value is not None:
                type_map = {
                    "str": str, "int": int, "float": (int, float),
                    "bool": bool, "list": list, "dict": dict,
                }
                expected = type_map.get(expected_type)
                if expected and not isinstance(value, expected):
                    errors.append(
                        f"{full_path}: expected {expected_type}, got {type(value).__name__}"
                    )

            if "children" in spec and isinstance(value, dict):
                self._validate_node(value, spec["children"], full_path, errors)

    def as_dict(self) -> Dict[str, Any]:
        """Return a deep copy of the full config."""
        with self._lock:
            return deepcopy(self._config)
