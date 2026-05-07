"""Serializable action schema for the offline decision/correction layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass
class _Action:
    action_type: str

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass
class GainAction(_Action):
    channel_id: str
    gain_db: float
    action_type: str = field(default="gain", init=False)


@dataclass
class EQAction(_Action):
    channel_id: str
    band_id: str
    freq_hz: float
    gain_db: float
    q: float
    filter_type: str = "peaking"
    action_type: str = field(default="eq", init=False)


@dataclass
class CompressorAction(_Action):
    channel_id: str
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_gain_db: float = 0.0
    action_type: str = field(default="compressor", init=False)


@dataclass
class GateExpanderAction(_Action):
    channel_id: str
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    action_type: str = field(default="gate_expander", init=False)


@dataclass
class PanAction(_Action):
    channel_id: str
    pan: float
    action_type: str = field(default="pan", init=False)


@dataclass
class FXSendAction(_Action):
    channel_id: str
    fx_bus: str
    send_db: float
    action_type: str = field(default="fx_send", init=False)


@dataclass
class NoChangeAction(_Action):
    channel_id: str = "mix"
    action_type: str = field(default="no_change", init=False)


DecisionAction = (
    GainAction
    | EQAction
    | CompressorAction
    | GateExpanderAction
    | PanAction
    | FXSendAction
    | NoChangeAction
)


@dataclass
class CandidateActionSet:
    """A candidate bundle proposed by rules or an optimizer."""

    candidate_id: str
    actions: list[DecisionAction] = field(default_factory=list)
    description: str = ""
    source: Literal["manual_rule", "nevergrad", "optuna", "fallback"] = "manual_rule"
    safety_limits_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "actions": [action.to_dict() for action in self.actions],
            "description": self.description,
            "source": self.source,
            "safety_limits_snapshot": _jsonable(self.safety_limits_snapshot),
            "metadata": _jsonable(self.metadata),
        }

    @property
    def is_no_change(self) -> bool:
        return not self.actions or all(action.action_type == "no_change" for action in self.actions)


def ensure_no_change_candidate(candidates: list[CandidateActionSet]) -> list[CandidateActionSet]:
    """Return candidates with a no-change baseline in the first slot."""

    if any(candidate.is_no_change for candidate in candidates):
        ordered = list(candidates)
        ordered.sort(key=lambda candidate: 0 if candidate.is_no_change else 1)
        return ordered
    return [
        CandidateActionSet(
            candidate_id="candidate_000_no_change",
            actions=[NoChangeAction()],
            description="Required no-change baseline.",
            source="manual_rule",
        ),
        *candidates,
    ]
