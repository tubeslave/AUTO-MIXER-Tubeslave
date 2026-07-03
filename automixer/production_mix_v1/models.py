"""Shared models for the opt-in production_mix_v1 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np


ACTION_GAIN = "gain"
ACTION_EQ = "eq"
ACTION_COMPRESSION = "compression"
ACTION_PARALLEL_COMPRESSION = "parallel_compression"
ACTION_FX_SEND = "fx_send"
ACTION_NOOP = "noop"

ACTION_STATUS_PLANNED = "planned"
ACTION_STATUS_RENDERED = "rendered"
ACTION_STATUS_VERIFIED = "verified"
ACTION_STATUS_FAILED = "failed"

MODE_OFFLINE = "offline"
MODE_LIVE = "live"

ROLE_ACTION_NO_OP = "no_op"
ROLE_ACTION_UNIVERSAL_SAFE = "universal_safe"
ROLE_ACTION_ROLE_SPECIFIC = "role_specific"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    return value


@dataclass(frozen=True)
class RoleDetectionResult:
    role: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False
    allowed_action_level: str = ROLE_ACTION_NO_OP

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass
class AudioStem:
    name: str
    audio: np.ndarray
    sample_rate: int
    role: str = "unknown"
    channel_id: int = 0
    current_fader_db: float = -12.0
    base_pan: float = 0.0
    path: str = ""
    role_detection: RoleDetectionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "channel_id": int(self.channel_id),
            "sample_rate": int(self.sample_rate),
            "samples": int(self.audio.shape[0]) if self.audio.ndim else 0,
            "path": self.path,
            "role_detection": self.role_detection.to_dict() if self.role_detection else None,
        }


@dataclass(frozen=True)
class MixAction:
    action_type: str
    target: str
    channel_id: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    stage: str = ""
    source_modules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class MixCandidate:
    name: str
    stage: str
    actions: list[MixAction] = field(default_factory=list)
    source_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return not self.actions

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class SafetyResult:
    candidate_name: str
    passed: bool
    allowed_actions: list[MixAction] = field(default_factory=list)
    blocked_actions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "passed": bool(self.passed),
            "allowed_actions": [a.to_dict() for a in self.allowed_actions],
            "blocked_actions": jsonable(self.blocked_actions),
            "warnings": list(self.warnings),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ActionRenderStatus:
    action: MixAction
    status: str = ACTION_STATUS_PLANNED
    rendered: bool = False
    verified: bool = False
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "status": self.status,
            "rendered": bool(self.rendered),
            "verified": bool(self.verified),
            "reason": self.reason,
            "metrics": jsonable(self.metrics),
        }


@dataclass(frozen=True)
class RenderResult:
    mix: np.ndarray
    processed_stems: dict[int, np.ndarray]
    action_statuses: list[ActionRenderStatus] = field(default_factory=list)

    def status_counts(self) -> dict[str, int]:
        return {
            "planned_actions_count": len(self.action_statuses),
            "rendered_actions_count": len([s for s in self.action_statuses if s.rendered]),
            "verified_actions_count": len([s for s in self.action_statuses if s.verified]),
            "failed_actions_count": len([s for s in self.action_statuses if s.status == ACTION_STATUS_FAILED]),
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: MixCandidate
    analysis: dict[str, Any]
    dimensions: dict[str, float]
    critic_scores: dict[str, Any]
    aggregate_score: float
    musical_score: float = 0.0
    technical_safety_score: float = 0.0
    final_score: float = 0.0
    score_delta: float = 0.0
    musical_score_delta: float = 0.0
    technical_score_delta: float = 0.0
    final_score_delta: float = 0.0
    muq_delta: float | None = None
    primary_evidence: list[str] = field(default_factory=list)
    proxy_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safety: SafetyResult | None = None
    action_statuses: list[dict[str, Any]] = field(default_factory=list)
    action_status_counts: dict[str, int] = field(default_factory=dict)
    render_verification: dict[str, Any] = field(default_factory=dict)
    critical_degradations: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    material_improvement: bool = False
    material_improvement_vs_no_change: bool = False
    material_improvement_vs_reference: bool = False

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class DecisionResult:
    selected: CandidateEvaluation
    baseline: CandidateEvaluation
    accepted: bool
    decision_state: str = "no_change"
    offline_render_accepted: bool = False
    console_actions_accepted: bool = False
    live_ready: bool = False
    selected_by_tiebreak: bool = False
    tiebreak_reason: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class OSCCommand:
    address: str
    args: list[Any] = field(default_factory=list)
    action: dict[str, Any] = field(default_factory=dict)
    dry_run_only: bool = True
    rationale: str = ""
    replay_correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class QueueFlushResult:
    dry_run: bool
    sent: list[dict[str, Any]] = field(default_factory=list)
    would_send: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: str
    mode: str
    dry_run: bool
    output_dir: str
    stems: list[dict[str, Any]]
    decision: DecisionResult
    queue_result: QueueFlushResult
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    report_fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = np.column_stack([array, array])
    elif array.ndim == 2 and array.shape[1] == 1:
        array = np.column_stack([array[:, 0], array[:, 0]])
    elif array.ndim == 2 and array.shape[1] > 2:
        array = array[:, :2]
    elif array.ndim == 0:
        array = np.zeros((0, 2), dtype=np.float32)
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def db_to_amp(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def amp_to_db(value: float, floor_db: float = -120.0) -> float:
    value = abs(float(value))
    if value <= 1e-12:
        return float(floor_db)
    return float(20.0 * np.log10(value))
