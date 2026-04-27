"""YAML-backed configuration for genre priors and metric thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


def _read_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def load_genre_profile(genre: str | None) -> Dict[str, Any]:
    """Load a genre profile, falling back to neutral priors."""
    profiles = _read_yaml("genre_profiles.yaml")
    key = (genre or "neutral").strip().lower().replace("_", "-")
    aliases = {
        "vocal-forward": "podcast",
        "podcast-like": "podcast",
        "electronic": "edm",
        "hiphop": "hip-hop",
    }
    key = aliases.get(key, key)
    profile = dict(profiles.get("neutral", {}))
    profile.update(dict(profiles.get(key, {})))
    profile["name"] = key if key in profiles else "neutral"
    profile["requested_name"] = genre or "neutral"
    return profile


def load_metric_thresholds() -> Dict[str, Any]:
    """Load default metric thresholds."""
    return _read_yaml("metric_thresholds.yaml")
