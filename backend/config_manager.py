"""
Configuration manager with YAML support and hot-reload via watchdog.
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
                'send_port': 2222,
                'receive_port': 2223,
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
                'llm_backend': 'auto',
                'llm_model': 'gpt-5.4',
                'ollama_url': 'http://localhost:11434',
                'openai_api_key': '',
                'openai_url': 'https://api.openai.com/v1/responses',
                'openai_reasoning_effort': 'low',
                'kimi_cli_path': '',
                'kimi_work_dir': '',
                'kimi_timeout_sec': 120,
                'model_fallbacks': [
                    'openai:gpt-5.4',
                    'kimi_cli:default',
                    'openai:gpt-4o-mini',
                ],
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
            'autofoh': {
                'classifier': {
                    'name_overrides': [],
                    'role_overrides': {},
                },
                'analysis': {
                    'fft_size': 4096,
                    'octave_fraction': 3,
                    'slope_compensation_db_per_octave': 4.5,
                },
                'detectors': {
                    'monitor_cycle_interval_sec': 1.0,
                    'lead_masking': {
                        'enabled': True,
                        'masking_threshold_db': 3.0,
                        'persistence_cycles': 3,
                        'min_culprit_contribution': 0.35,
                        'lead_boost_db': 0.5,
                    },
                    'mud_excess': {
                        'enabled': True,
                        'threshold_db': 2.5,
                        'persistence_cycles': 3,
                        'hysteresis_db': 0.75,
                    },
                    'harshness_excess': {
                        'enabled': True,
                        'threshold_db': 2.5,
                        'persistence_cycles': 3,
                        'hysteresis_db': 0.75,
                    },
                    'sibilance_excess': {
                        'enabled': True,
                        'threshold_db': 2.5,
                        'persistence_cycles': 3,
                        'hysteresis_db': 0.75,
                    },
                    'low_end': {
                        'enabled': True,
                        'sub_threshold_db': 4.0,
                        'bass_threshold_db': 3.0,
                        'body_threshold_db': 2.5,
                        'persistence_cycles': 3,
                        'hysteresis_db': 0.75,
                        'min_culprit_contribution': 0.35,
                    },
                },
                'evaluation': {
                    'enabled': True,
                    'evaluation_window_sec': 2.0,
                    'allow_proxy_audio_evaluation_for_testing': False,
                    'allow_proxy_audio_rollback_for_testing': False,
                    'min_band_improvement_db': 0.25,
                    'min_rms_response_db': 0.1,
                    'worsening_tolerance_db': 0.5,
                },
                'logging': {
                    'enabled': False,
                    'path': '',
                    'queue_maxsize': 1024,
                    'write_session_report_on_stop': True,
                    'report_path': '',
                },
                'soundcheck_profile': {
                    'enabled': True,
                    'auto_save_after_analysis': True,
                    'auto_load_on_start': True,
                    'capture_multiphase_learning': True,
                    'silence_capture_duration_sec': 0.75,
                    'path': '',
                    'use_loaded_target_corridor': True,
                    'use_phase_target_action_guards': True,
                    'replace_live_target_with_learned_corridor': True,
                },
                'safety': {
                    'minimum_auto_apply_classification_confidence': 0.75,
                    'new_or_unknown_channel_auto_corrections_enabled': False,
                    'action_limits': {
                        'channel_fader_max_step_db': 1.0,
                        'channel_fader_min_interval_sec': 3.0,
                        'lead_fader_max_step_db': 0.5,
                        'lead_fader_min_interval_sec': 2.0,
                        'gain_max_abs_db': 12.0,
                        'gain_min_interval_sec': 3.0,
                        'broad_eq_max_step_db': 1.0,
                        'broad_eq_max_total_db_from_snapshot': 3.0,
                        'broad_eq_min_interval_sec': 5.0,
                        'feedback_notch_max_cut_db': -6.0,
                        'feedback_notch_min_q': 8.0,
                        'feedback_notch_ttl_sec': 120.0,
                    },
                },
            },
            'perceptual': {
                'enabled': False,
                'mode': 'shadow',
                'backend': 'lightweight',
                'model_name': 'm-a-p/MERT-v1-95M',
                'sample_rate': 24000,
                'window_seconds': 5,
                'hop_seconds': 2,
                'evaluate_channels': True,
                'evaluate_mix_bus': True,
                'max_cpu_percent': 25,
                'log_scores': True,
                'block_osc_when_score_worse': False,
                'log_path': 'logs/perceptual_decisions.jsonl',
                'queue_maxsize': 128,
                'async_evaluation': True,
                'improvement_threshold': 0.03,
                'local_files_only': False,
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
            'AUTOMIXER_LLM_MODEL': ('ai', 'llm_model'),
            'OPENAI_API_KEY': ('ai', 'openai_api_key'),
            'AUTOMIXER_OPENAI_API_KEY': ('ai', 'openai_api_key'),
            'KIMI_CLI_PATH': ('ai', 'kimi_cli_path'),
            'AUTOMIXER_KIMI_CLI': ('ai', 'kimi_cli_path'),
            'AUTOMIXER_KIMI_WORK_DIR': ('ai', 'kimi_work_dir'),
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

        fallbacks = os.environ.get('AUTOMIXER_MODEL_FALLBACKS')
        if fallbacks is not None:
            self._config['ai']['model_fallbacks'] = [
                item.strip()
                for item in fallbacks.split(',')
                if item.strip()
            ]

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
