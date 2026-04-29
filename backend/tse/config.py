"""Configuration helpers for experimental streaming TSE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TSEConfig:
    enabled: bool = False
    mode: str = "bypass"
    conservative_mode: bool = True
    max_latency_ms: float = 20.0
    chunk_size_ms: float = 10.0
    lookback_chunks: int = 8
    min_confidence_for_control: float = 0.65
    use_for_analysis_only: bool = True
    fallback_to_original: bool = True
    low_confidence_bleed_threshold: float = 0.55
    log_interval_sec: float = 10.0
    per_instrument: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any] | None = None) -> "TSEConfig":
        payload = dict(payload or {})
        per_instrument = dict(payload.pop("per_instrument", {}) or {})
        config = cls(**{key: value for key, value in payload.items() if key in cls.__dataclass_fields__})
        config.per_instrument = per_instrument or default_per_instrument()
        return config

    def instrument_enabled(self, instrument_type: str | None) -> bool:
        if not self.enabled:
            return False
        if not instrument_type:
            return True
        normalized = normalize_instrument(instrument_type)
        settings = self.per_instrument.get(normalized)
        if settings is None:
            return False
        return bool(settings.get("enabled", False))


def normalize_instrument(instrument_type: str | None) -> str:
    value = str(instrument_type or "").strip()
    if not value:
        return "unknown"
    compact = value.replace("-", "_").replace(" ", "_")
    lower = compact.lower()
    if lower in {"leadvocal", "lead_vocal", "backvocal", "backing_vocal", "vocal", "vocals"}:
        return "vocal"
    if lower in {"kick_in", "kick_out"}:
        return "kick"
    if lower in {"snare_top", "snare_bottom"}:
        return "snare"
    if "tom" in lower:
        return "tom"
    if "guitar" in lower:
        return "guitar"
    if "bass" in lower:
        return "bass"
    if lower in {"hihat", "hi_hat", "ride", "overhead", "overheads", "cymbals"}:
        return "cymbals"
    return lower


def default_per_instrument() -> Dict[str, Dict[str, Any]]:
    return {
        "vocal": {"enabled": True},
        "kick": {"enabled": True},
        "snare": {"enabled": True},
        "tom": {"enabled": True},
        "guitar": {"enabled": False},
        "bass": {"enabled": False},
        "cymbals": {"enabled": False},
        "unknown": {"enabled": False},
    }
