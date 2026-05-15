"""Typed v2 decision models.

The Decision Engine produces these models only. It does not know how to send
OSC, MIDI, or any mixer-specific control message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping


ACTION_GAIN = "gain_correction"
ACTION_EQ = "eq_correction"
ACTION_COMPRESSION = "compression_correction"
ACTION_PAN = "pan_correction"
ACTION_NO_ACTION = "no_action"

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"


def jsonable(value: Any) -> Any:
    """Convert common Python/DSP values into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        return {str(key): jsonable(item) for key, item in value.__dict__.items()}
    return repr(value)


def clamp_confidence(value: float) -> float:
    """Clamp confidence to the public 0..1 range."""
    return round(max(0.0, min(1.0, float(value))), 3)


@dataclass(frozen=True)
class ActionDecision:
    """One proposed correction or explicit no-op."""

    id: str
    action_type: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    risk_level: str = RISK_MEDIUM
    source_modules: List[str] = field(default_factory=list)
    expected_audio_effect: str = ""
    safe_to_apply: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = clamp_confidence(payload["confidence"])
        return jsonable(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionDecision":
        return cls(
            id=str(payload.get("id", "")),
            action_type=str(payload.get("action_type", ACTION_NO_ACTION)),
            target=str(payload.get("target", "mix")),
            parameters=dict(payload.get("parameters", {})),
            reason=str(payload.get("reason", "")),
            confidence=clamp_confidence(float(payload.get("confidence", 0.0))),
            risk_level=str(payload.get("risk_level", RISK_MEDIUM)),
            source_modules=[str(item) for item in payload.get("source_modules", [])],
            expected_audio_effect=str(payload.get("expected_audio_effect", "")),
            safe_to_apply=bool(payload.get("safe_to_apply", False)),
            metadata=dict(payload.get("metadata", {})),
        )

    def with_updates(self, **updates: Any) -> "ActionDecision":
        return replace(self, **updates)


@dataclass(frozen=True)
class ActionPlan:
    """A batch of decisions for one analysis pass."""

    plan_id: str
    created_at: float = field(default_factory=time.time)
    mode: str = "live"
    decisions: List[ActionDecision] = field(default_factory=list)
    source_modules: List[str] = field(default_factory=list)
    input_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def safe_decisions(self) -> List[ActionDecision]:
        return [
            decision
            for decision in self.decisions
            if decision.safe_to_apply and decision.action_type != ACTION_NO_ACTION
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at": float(self.created_at),
            "mode": self.mode,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "source_modules": list(self.source_modules),
            "input_summary": jsonable(self.input_summary),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionPlan":
        return cls(
            plan_id=str(payload.get("plan_id", "")),
            created_at=float(payload.get("created_at", time.time())),
            mode=str(payload.get("mode", "live")),
            decisions=[
                ActionDecision.from_dict(item)
                for item in payload.get("decisions", [])
                if isinstance(item, Mapping)
            ],
            source_modules=[str(item) for item in payload.get("source_modules", [])],
            input_summary=dict(payload.get("input_summary", {})),
            notes=[str(item) for item in payload.get("notes", [])],
        )
