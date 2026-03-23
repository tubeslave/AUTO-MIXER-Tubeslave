"""Configuration loading and management for AutoMixer backend."""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Handles loading and saving configuration from files.
    """

    def __init__(self, config_dir: str = None):
        """Initialize ConfigLoader with path to config directory."""
        if config_dir is None:
            config_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "config"
            )
        self.config_dir = config_dir
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from default_config.json"""
        config_path = os.path.join(self.config_dir, "default_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return self._config
        except Exception as e:
            logger.warning(
                f"Failed to load config from {config_path}: {e}. Using defaults."
            )
            self._config = {
                "automation": {
                    "auto_gain": {
                        "bleeding_rejection": {
                            "enabled": True,
                            "correlation_threshold": 0.7,
                            "level_difference_threshold_db": 8.0,
                        }
                    }
                }
            }
            return self._config

    @property
    def config(self) -> dict:
        """Return the loaded configuration."""
        return self._config

    def get_user_config_path(self) -> str:
        """Get path to user config file."""
        return os.path.join(self.config_dir, "user_config.json")

    def load_user_config(self) -> dict:
        """Load user-saved settings from user_config.json."""
        path = self.get_user_config_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"User config loaded from {path}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load user config: {e}")
            return {}

    def save_user_config(self, section: str, settings: dict):
        """Save user settings to user_config.json under a given section."""
        path = self.get_user_config_path()
        # Load existing user config
        existing = self.load_user_config()
        existing[section] = settings
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            logger.info(f"User config saved to {path}: section={section}")
        except Exception as e:
            logger.error(f"Failed to save user config: {e}")
            raise

    def get_method_preset_file(self, method_name: str, fallback: str) -> str:
        """Resolve method preset base file from config."""
        try:
            return (
                self._config.get("automation", {})
                .get("preset_bases", {})
                .get(method_name, {})
                .get("file", fallback)
            )
        except Exception:
            return fallback
