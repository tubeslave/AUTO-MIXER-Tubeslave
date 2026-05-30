"""Typed schemas for concert/soundcheck learning telemetry."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


SCHEMA_VERSION = "0.1"
SESSION_MODES = {"soundcheck", "concert", "rehearsal", "offline_test"}
ACTORS = {"human", "automixer", "test"}
TARGET_TYPES = {"channel", "bus", "master", "scene"}
ACTION_TYPES = {
    "fader_move",
    "trim_change",
    "mute",
    "unmute",
    "eq_change",
    "comp_change",
    "gate_change",
    "snapshot_recall",
    "note",
}
LABEL_SOURCES = {"human", "agent", "evaluator"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def generate_session_id(mode: str = "soundcheck") -> str:
    safe_mode = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(mode or "session")).strip("-").lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{safe_mode}_{uuid4().hex[:8]}"


def _require_enum(value: str, allowed: set[str], field_name: str) -> str:
    normalized = str(value or "").strip()
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return normalized


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value, got {value!r}") from exc


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer value, got {value!r}") from exc


def _asdict_clean(value: Any) -> Dict[str, Any]:
    return asdict(value)


@dataclass
class SessionRecord:
    session_id: str
    started_at: str
    mode: str
    ended_at: Optional[str] = None
    venue_name: str = ""
    artist_name: str = ""
    project_version: str = ""
    console_type: str = ""
    console_ip: str = ""
    safety_mode: str = "dry-run"
    notes: str = ""

    def __post_init__(self) -> None:
        self.mode = _require_enum(self.mode, SESSION_MODES, "mode")
        if not self.session_id:
            raise ValueError("session_id is required")

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class SceneMetadata:
    venue_name: str = ""
    room_type: str = ""
    estimated_capacity: Optional[int] = None
    pa_type: str = ""
    subs_mode: str = ""
    monitor_type: str = ""
    routing_notes: str = ""
    weather_notes: str = ""
    engineer_notes: str = ""

    def __post_init__(self) -> None:
        self.estimated_capacity = _optional_int(self.estimated_capacity)

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class ChannelTelemetry:
    timestamp_ms: int
    session_id: str
    channel_id: int
    channel_name: str = ""
    detected_instrument: str = ""
    fader_db: Optional[float] = None
    trim_db: Optional[float] = None
    mute: Optional[bool] = None
    solo: Optional[bool] = None
    pan: Optional[float] = None
    meter_rms_dbfs: Optional[float] = None
    meter_peak_dbfs: Optional[float] = None
    gate_gain_reduction_db: Optional[float] = None
    comp_gain_reduction_db: Optional[float] = None
    eq_summary: Dict[str, Any] = field(default_factory=dict)
    plugin_or_processing_snapshot: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp_ms = int(self.timestamp_ms)
        self.channel_id = int(self.channel_id)
        for field_name in (
            "fader_db",
            "trim_db",
            "pan",
            "meter_rms_dbfs",
            "meter_peak_dbfs",
            "gate_gain_reduction_db",
            "comp_gain_reduction_db",
        ):
            setattr(self, field_name, _optional_float(getattr(self, field_name)))

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class BusTelemetry:
    timestamp_ms: int
    session_id: str
    bus_name: str
    rms_dbfs: Optional[float] = None
    peak_dbfs: Optional[float] = None
    lufs_short: Optional[float] = None
    limiter_gain_reduction_db: Optional[float] = None

    def __post_init__(self) -> None:
        self.timestamp_ms = int(self.timestamp_ms)
        for field_name in ("rms_dbfs", "peak_dbfs", "lufs_short", "limiter_gain_reduction_db"):
            setattr(self, field_name, _optional_float(getattr(self, field_name)))

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class SPLTelemetry:
    timestamp_ms: int
    session_id: str
    laeq_1s: Optional[float] = None
    lcpeak: Optional[float] = None
    lafmax: Optional[float] = None
    source: str = "mock"
    calibration_notes: str = ""

    def __post_init__(self) -> None:
        self.timestamp_ms = int(self.timestamp_ms)
        for field_name in ("laeq_1s", "lcpeak", "lafmax"):
            setattr(self, field_name, _optional_float(getattr(self, field_name)))

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class EngineerAction:
    timestamp_ms: int
    session_id: str
    actor: str
    target_type: str
    target_id: str
    action_type: str
    before_value: Any = None
    after_value: Any = None
    reason: str = ""
    confidence: Optional[float] = None
    safety_status: str = "logged_only"

    def __post_init__(self) -> None:
        self.timestamp_ms = int(self.timestamp_ms)
        self.actor = _require_enum(self.actor, ACTORS, "actor")
        self.target_type = _require_enum(self.target_type, TARGET_TYPES, "target_type")
        self.action_type = _require_enum(self.action_type, ACTION_TYPES, "action_type")
        self.confidence = _optional_float(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class LabelAnnotation:
    timestamp_ms: int
    session_id: str
    label: str
    severity: str = "info"
    confidence: Optional[float] = None
    note: str = ""
    source: str = "human"

    def __post_init__(self) -> None:
        self.timestamp_ms = int(self.timestamp_ms)
        self.source = _require_enum(self.source, LABEL_SOURCES, "source")
        self.confidence = _optional_float(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


@dataclass
class TelemetryManifest:
    session_id: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    files: Dict[str, str] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    project_git_commit: Optional[str] = None
    safety_mode: str = "dry-run"

    def to_dict(self) -> Dict[str, Any]:
        return _asdict_clean(self)


EVENT_FILES = {
    "channel_telemetry": "channel_telemetry.jsonl",
    "bus_telemetry": "bus_telemetry.jsonl",
    "spl_telemetry": "spl_telemetry.jsonl",
    "engineer_actions": "engineer_actions.jsonl",
    "labels": "labels.jsonl",
}
