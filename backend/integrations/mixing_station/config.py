"""Configuration loading for the Mixing Station adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str | Path | None, *, base: Path = REPO_ROOT) -> Path:
    """Resolve repository-relative paths used by config files."""
    if not path:
        return base
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base / candidate


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """Load YAML as a dict, returning an empty dict for missing files."""
    resolved = resolve_repo_path(path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class MixingStationConfig:
    """Runtime config for Mixing Station visualization/control."""

    enabled: bool = False
    transport: str = "websocket"
    host: str = "127.0.0.1"
    rest_port: int = 8080
    osc_port: int = 9000
    console_profile: str = "wing_rack"
    mode: str = "offline_visualization"
    dry_run: bool = True
    live_control: bool = False
    discover_paths_on_startup: bool = False
    log_jsonl: str = "logs/automix_to_mixing_station.jsonl"
    capabilities_file: str = ""
    mapping_file: str = ""
    safety_file: str = "config/mixing_station/safety.yaml"
    emergency_stop_file: str = "runtime/EMERGENCY_STOP"
    rest_command_endpoint: str = ""
    websocket_path: str = "/ws"
    websocket_command_path_template: str = ""
    rest_app_state_paths: List[str] = field(
        default_factory=lambda: ["/app/state", "/api/app/state", "/api/v1/app/state"]
    )
    rest_mixer_state_paths: List[str] = field(
        default_factory=lambda: ["/mixer/state", "/api/mixer/state", "/api/v1/mixer/state"]
    )
    rest_discovery_paths: List[str] = field(
        default_factory=lambda: [
            "/console/data/paths",
            "/api/explorer",
            "/api/paths",
            "/api/v1/paths",
            "/openapi.json",
        ]
    )

    @classmethod
    def from_file(cls, path: str | Path = "config/mixing_station.yaml") -> "MixingStationConfig":
        """Load adapter config from YAML."""
        data = load_yaml_file(path)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]] = None) -> "MixingStationConfig":
        """Build config from a mapping, filling profile-specific defaults."""
        data = dict(data or {})
        profile = normalize_console_profile(str(data.get("console_profile", "wing_rack")))
        default_capabilities = f"config/mixing_station/{profile}_capabilities.yaml"
        default_mapping = f"config/mixing_station/maps/{profile}.yaml"
        return cls(
            enabled=bool(data.get("enabled", False)),
            transport=str(data.get("transport", "websocket")).lower(),
            host=str(data.get("host", "127.0.0.1")),
            rest_port=int(data.get("rest_port", 8080)),
            osc_port=int(data.get("osc_port", 9000)),
            console_profile=profile,
            mode=str(data.get("mode", "offline_visualization")),
            dry_run=bool(data.get("dry_run", True)),
            live_control=bool(data.get("live_control", False)),
            discover_paths_on_startup=bool(data.get("discover_paths_on_startup", False)),
            log_jsonl=str(data.get("log_jsonl", "logs/automix_to_mixing_station.jsonl")),
            capabilities_file=str(data.get("capabilities_file") or default_capabilities),
            mapping_file=str(data.get("mapping_file") or default_mapping),
            safety_file=str(data.get("safety_file", "config/mixing_station/safety.yaml")),
            emergency_stop_file=str(data.get("emergency_stop_file", "runtime/EMERGENCY_STOP")),
            rest_command_endpoint=str(data.get("rest_command_endpoint", "")),
            websocket_path=str(data.get("websocket_path", "/ws")),
            websocket_command_path_template=str(
                data.get("websocket_command_path_template", "")
            ),
            rest_app_state_paths=list(
                data.get(
                    "rest_app_state_paths",
                    ["/app/state", "/api/app/state", "/api/v1/app/state"],
                )
            ),
            rest_mixer_state_paths=list(
                data.get(
                    "rest_mixer_state_paths",
                    ["/mixer/state", "/api/mixer/state", "/api/v1/mixer/state"],
                )
            ),
            rest_discovery_paths=list(
                data.get(
                    "rest_discovery_paths",
                    [
                        "/console/data/paths",
                        "/api/explorer",
                        "/api/paths",
                        "/api/v1/paths",
                        "/openapi.json",
                    ],
                )
            ),
        )

    def path(self, value: str | Path) -> Path:
        """Resolve a path relative to the repository root."""
        return resolve_repo_path(value)


def normalize_console_profile(value: str) -> str:
    """Normalize user/profile aliases to internal profile ids."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "wing": "wing_rack",
        "wingrack": "wing_rack",
        "wing_rack": "wing_rack",
        "behringer_wing": "wing_rack",
        "behringer_wing_rack": "wing_rack",
        "dlive": "dlive",
        "d_live": "dlive",
        "allen_heath_dlive": "dlive",
        "allen_&_heath_dlive": "dlive",
    }
    return aliases.get(normalized, normalized)
