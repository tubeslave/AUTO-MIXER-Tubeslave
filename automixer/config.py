"""Config loader for Decision Engine v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from automixer.decision import DecisionEngineConfig
from automixer.safety import SafetyGateConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "automixer.yaml"


@dataclass(frozen=True)
class DecisionEngineV2RuntimeConfig:
    """Top-level v2 runtime config."""

    enabled: bool = False
    dry_run: bool = True
    offline_experiment: bool = False
    knowledge_path: str = "automixer/knowledge/mixing_knowledge_base.json"
    decision: DecisionEngineConfig = field(default_factory=DecisionEngineConfig)
    safety: SafetyGateConfig = field(default_factory=lambda: SafetyGateConfig(dry_run=True))
    log_path: str = "~/Desktop/Ai LOGS/decision_engine_v2.jsonl"


def load_decision_engine_v2_config(
    config_path: str | Path | None = None,
) -> DecisionEngineV2RuntimeConfig:
    """Load ``decision_engine_v2`` from `config/automixer.yaml`."""
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    payload = _load_mapping(path)
    section = dict(payload.get("decision_engine_v2", {}) or {})
    dry_run = bool(section.get("dry_run", True))
    safety_mapping = dict(section.get("safety", {}) or {})
    safety_mapping.setdefault("dry_run", dry_run)
    return DecisionEngineV2RuntimeConfig(
        enabled=bool(section.get("enabled", False)),
        dry_run=dry_run,
        offline_experiment=bool(section.get("offline_experiment", False)),
        knowledge_path=str(
            section.get("knowledge_path", "automixer/knowledge/mixing_knowledge_base.json")
        ),
        decision=DecisionEngineConfig.from_mapping(section.get("decision", {})),
        safety=SafetyGateConfig.from_mapping(safety_mapping),
        log_path=str(section.get("log_path", "~/Desktop/Ai LOGS/decision_engine_v2.jsonl")),
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except Exception:
        return {}
