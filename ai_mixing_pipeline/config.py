"""Configuration loading for AI mixing roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_WEIGHTS = {
    "muq_eval": 0.30,
    "audiobox_aesthetics": 0.20,
    "mert": 0.15,
    "clap": 0.10,
    "essentia": 0.10,
    "panns_or_beats": 0.10,
    "safety": 0.05,
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load ai_mixing_roles.yaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def load_roles_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load roles config and fill conservative defaults."""

    config_path = Path(path or "configs/ai_mixing_roles.yaml").expanduser()
    payload = _load_yaml(config_path)
    critics = dict(payload.get("critics", {}) or {})
    for name, weight in DEFAULT_WEIGHTS.items():
        if name == "safety":
            continue
        section = dict(critics.get(name, {}) or {})
        section.setdefault("enabled", True)
        section.setdefault("weight", weight)
        section.setdefault("role", name)
        critics[name] = section
    payload["critics"] = critics
    payload.setdefault("safety", {})
    payload.setdefault("offline_test", {})
    payload["offline_test"].setdefault("input_dir", "offline_test_input")
    payload["offline_test"].setdefault("output_dir", "offline_test_output")
    payload["offline_test"].setdefault("create_no_change_candidate", True)
    payload["offline_test"].setdefault("loudness_match_candidates", True)
    payload["offline_test"].setdefault("safe_render_peak_margin_db", 0.6)
    payload["offline_test"].setdefault("safe_render_peak_ceiling_dbfs", None)
    payload["offline_test"].setdefault("save_all_renders", True)
    payload["offline_test"].setdefault("save_reports", True)
    return payload


def enabled_critic_weights(config: dict[str, Any]) -> dict[str, float]:
    """Return configured weights for enabled critic sections."""

    weights: dict[str, float] = {}
    for name, section in (config.get("critics", {}) or {}).items():
        if not isinstance(section, dict) or not bool(section.get("enabled", True)):
            continue
        if name in {"demucs_or_openunmix"}:
            continue
        weights[name] = float(section.get("weight", DEFAULT_WEIGHTS.get(name, 0.0)))
    weights["safety"] = float((config.get("safety", {}) or {}).get("weight", DEFAULT_WEIGHTS["safety"]))
    if weights["safety"] <= 0.0:
        weights["safety"] = DEFAULT_WEIGHTS["safety"]
    return weights
