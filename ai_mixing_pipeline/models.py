"""Shared data models for the offline AI mixing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def jsonable(value: Any) -> Any:
    """Convert common project values to JSON-safe structures."""

    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


@dataclass
class AudioTrack:
    """Loaded audio file plus inferred role metadata."""

    name: str
    path: str
    role: str = "unknown"
    sample_rate: int = 0
    duration_sec: float = 0.0
    channels: int = 1

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class CandidateAction:
    """One bounded engineering action proposed for an offline candidate."""

    action_type: str
    target: str = "mix"
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    source: str = "ai_mixing_pipeline"
    safe_range: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class MixCandidate:
    """A named candidate render and its proposed actions."""

    candidate_id: str
    label: str
    actions: list[CandidateAction] = field(default_factory=list)
    render_filename: str = ""
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class RenderResult:
    """Output of a sandbox render pass."""

    candidate_id: str
    path: str
    sample_rate: int
    duration_sec: float
    loudness_matched: bool = False
    output_gain_db: float = 0.0
    audit: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class SafetyResult:
    """Safety Governor verdict for one candidate."""

    candidate_id: str
    passed: bool
    safety_score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass
class DecisionResult:
    """Decision Engine output for candidate selection."""

    selected_candidate_id: str
    final_scores: dict[str, float]
    normalized_weights: dict[str, float]
    explanations: dict[str, str]
    no_change_selected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)
