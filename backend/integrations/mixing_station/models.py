"""Shared data models for Automixer to Mixing Station correction mirroring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


CorrectionValue = float | int | bool | str


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any) -> datetime:
    """Parse timestamps used in AutomixCorrection payloads."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), timezone.utc)
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return utc_now()


@dataclass
class AutomixCorrection:
    """A normalized correction decision produced by the Automixer."""

    console_profile: str
    mode: str
    channel_index: int
    parameter: str
    value: CorrectionValue
    value_unit: str
    reason: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=utc_now)
    channel_name: Optional[str] = None
    strip_type: str = "input"
    previous_value: Optional[CorrectionValue] = None
    confidence: float = 1.0
    source_metrics: Dict[str, Any] = field(default_factory=dict)
    safety_status: str = "allowed"
    dry_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "console_profile": self.console_profile,
            "mode": self.mode,
            "channel_index": self.channel_index,
            "channel_name": self.channel_name,
            "strip_type": self.strip_type,
            "parameter": self.parameter,
            "value": self.value,
            "value_unit": self.value_unit,
            "previous_value": self.previous_value,
            "reason": self.reason,
            "confidence": self.confidence,
            "source_metrics": dict(self.source_metrics),
            "safety_status": self.safety_status,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AutomixCorrection":
        """Create a correction from a JSON/dict payload."""
        return cls(
            id=str(payload.get("id") or uuid4()),
            timestamp=parse_timestamp(payload.get("timestamp")),
            console_profile=str(payload.get("console_profile", "wing_rack")),
            mode=str(payload.get("mode", "offline_visualization")),
            channel_index=int(payload.get("channel_index", 0)),
            channel_name=payload.get("channel_name"),
            strip_type=str(payload.get("strip_type", "input")),
            parameter=str(payload.get("parameter", "fader")),
            value=payload.get("value"),
            value_unit=str(payload.get("value_unit", payload.get("unit", "db"))),
            previous_value=payload.get("previous_value"),
            reason=str(payload.get("reason", "")),
            confidence=float(payload.get("confidence", 1.0)),
            source_metrics=dict(payload.get("source_metrics") or {}),
            safety_status=str(payload.get("safety_status", "allowed")),
            dry_run=bool(payload.get("dry_run", True)),
        )


@dataclass(frozen=True)
class MixingStationCommand:
    """Transport-ready command mapped to a Mixing Station dataPath."""

    transport: str
    data_path: str
    value: CorrectionValue
    value_format: str = "plain"
    method: str = "SET"
    parameter: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transport": self.transport,
            "data_path": self.data_path,
            "format": self.value_format,
            "method": self.method,
            "parameter": self.parameter,
            "value": self.value,
        }


@dataclass
class CorrectionResult:
    """Result of validating, logging, and optionally sending a correction."""

    correction: AutomixCorrection
    success: bool
    dry_run: bool
    sent: bool = False
    transport: str = ""
    data_path: Optional[str] = None
    sent_value: Optional[CorrectionValue] = None
    requested_value: Optional[CorrectionValue] = None
    safety_status: str = "allowed"
    error: Optional[str] = None
    message: str = ""
    blocked: bool = False
    command: Optional[MixingStationCommand] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correction": self.correction.to_dict(),
            "success": self.success,
            "dry_run": self.dry_run,
            "sent": self.sent,
            "transport": self.transport,
            "data_path": self.data_path,
            "sent_value": self.sent_value,
            "requested_value": self.requested_value,
            "safety_status": self.safety_status,
            "error": self.error,
            "message": self.message,
            "blocked": self.blocked,
            "command": self.command.to_dict() if self.command else None,
        }
