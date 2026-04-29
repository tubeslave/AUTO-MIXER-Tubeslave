"""Shared models for `production_mix_v1`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

try:
    from automixer.decision.models import jsonable
except Exception:
    def jsonable(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [jsonable(item) for item in value]
        if hasattr(value, "to_dict"):
            return jsonable(value.to_dict())
        return value


ACTION_GAIN = "gain"
ACTION_EQ = "eq"
ACTION_COMPRESSION = "compression"
ACTION_PAN = "pan"
ACTION_FX_SEND = "fx_send"
ACTION_PARALLEL_COMPRESSION = "parallel_compression"
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


@dataclass(frozen=True)
class RoleDetectionResult:
    """Evidence-backed source role classification for one input stem."""

    role: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False
    allowed_action_level: str = ROLE_ACTION_NO_OP

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "confidence": float(self.confidence),
            "evidence": jsonable(self.evidence),
            "is_ambiguous": bool(self.is_ambiguous),
            "allowed_action_level": self.allowed_action_level,
        }


@dataclass
class AudioStem:
    """One source/stem used by the offline candidate renderer."""

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
            "channels": int(self.audio.shape[1]) if self.audio.ndim == 2 else 1,
            "current_fader_db": float(self.current_fader_db),
            "base_pan": float(self.base_pan),
            "path": self.path,
            "role_detection": self.role_detection.to_dict() if self.role_detection else None,
        }


@dataclass(frozen=True)
class MixAction:
    """A candidate action before final Safety Governor validation."""

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
    """A full render candidate with zero or more staged actions."""

    name: str
    stage: str
    actions: list[MixAction] = field(default_factory=list)
    source_modules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return not self.actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "actions": [action.to_dict() for action in self.actions],
            "source_modules": list(self.source_modules),
            "metadata": jsonable(self.metadata),
        }


@dataclass(frozen=True)
class SafetyResult:
    """Safety Governor result for one candidate."""

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
            "allowed_actions": [action.to_dict() for action in self.allowed_actions],
            "blocked_actions": jsonable(self.blocked_actions),
            "warnings": list(self.warnings),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class ActionRenderStatus:
    """Offline renderer proof for one planned action."""

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
    """Rendered candidate audio plus proof of applied actions."""

    mix: np.ndarray
    processed_stems: dict[int, np.ndarray]
    action_statuses: list[ActionRenderStatus] = field(default_factory=list)

    @property
    def planned_count(self) -> int:
        return len(self.action_statuses)

    @property
    def rendered_count(self) -> int:
        return len([item for item in self.action_statuses if item.rendered])

    @property
    def verified_count(self) -> int:
        return len([item for item in self.action_statuses if item.verified])

    @property
    def failed_count(self) -> int:
        return len([item for item in self.action_statuses if item.status == ACTION_STATUS_FAILED])

    def status_counts(self) -> dict[str, int]:
        return {
            "planned_actions_count": self.planned_count,
            "rendered_actions_count": self.rendered_count,
            "verified_actions_count": self.verified_count,
            "failed_actions_count": self.failed_count,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """Scores and safety evidence for a candidate."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "analysis": jsonable(self.analysis),
            "dimensions": jsonable(self.dimensions),
            "critic_scores": jsonable(self.critic_scores),
            "aggregate_score": float(self.aggregate_score),
            "musical_score": float(self.musical_score),
            "technical_safety_score": float(self.technical_safety_score),
            "final_score": float(self.final_score),
            "score_delta": float(self.score_delta),
            "musical_score_delta": float(self.musical_score_delta),
            "technical_score_delta": float(self.technical_score_delta),
            "final_score_delta": float(self.final_score_delta),
            "muq_delta": self.muq_delta,
            "primary_evidence": list(self.primary_evidence),
            "proxy_evidence": list(self.proxy_evidence),
            "warnings": list(self.warnings),
            "safety": self.safety.to_dict() if self.safety else None,
            "action_statuses": jsonable(self.action_statuses),
            "action_status_counts": jsonable(self.action_status_counts),
            "render_verification": jsonable(self.render_verification),
            "critical_degradations": list(self.critical_degradations),
            "rationale": list(self.rationale),
            "material_improvement": bool(self.material_improvement),
        }


@dataclass(frozen=True)
class DecisionResult:
    """Final candidate selection before command queue flushing."""

    selected: CandidateEvaluation
    baseline: CandidateEvaluation
    accepted: bool
    decision_state: str = "no_change"
    offline_render_accepted: bool = False
    console_actions_accepted: bool = False
    live_ready: bool = False
    rejected: list[dict[str, Any]] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "baseline": self.baseline.to_dict(),
            "accepted": bool(self.accepted),
            "decision_state": self.decision_state,
            "offline_render_accepted": bool(self.offline_render_accepted),
            "console_actions_accepted": bool(self.console_actions_accepted),
            "live_ready": bool(self.live_ready),
            "rejected": jsonable(self.rejected),
            "rationale": list(self.rationale),
        }


@dataclass(frozen=True)
class OSCCommand:
    """Queued OSC command. Queueing is not sending."""

    address: str
    args: list[Any] = field(default_factory=list)
    action: dict[str, Any] = field(default_factory=dict)
    dry_run_only: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class QueueFlushResult:
    """Result of flushing or dry-running an OSC queue."""

    dry_run: bool
    sent: list[dict[str, Any]] = field(default_factory=list)
    would_send: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass(frozen=True)
class PipelineRunResult:
    """Top-level pipeline result."""

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
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "dry_run": bool(self.dry_run),
            "output_dir": self.output_dir,
            "stems": jsonable(self.stems),
            "decision": self.decision.to_dict(),
            "queue_result": self.queue_result.to_dict(),
            "evaluations": jsonable(self.evaluations),
            "artifacts": dict(self.artifacts),
            "report_fields": jsonable(self.report_fields),
            "warnings": list(self.warnings),
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
        }


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Return finite float32 audio shaped `(samples, 2)`."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        array = np.zeros((0, 2), dtype=np.float32)
    elif array.ndim == 1:
        array = np.column_stack([array, array])
    elif array.ndim == 2:
        if array.shape[0] == 2 and array.shape[1] > 2:
            array = array.T
        if array.shape[1] == 1:
            array = np.column_stack([array[:, 0], array[:, 0]])
        elif array.shape[1] > 2:
            array = array[:, :2]
    else:
        array = array.reshape(-1, 1)
        array = np.column_stack([array[:, 0], array[:, 0]])
    return np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def db_to_amp(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def amp_to_db(value: float, floor_db: float = -120.0) -> float:
    value = abs(float(value))
    if value <= 1e-12:
        return float(floor_db)
    return float(20.0 * np.log10(value))


def candidate_artifact_name(name: str, suffix: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return f"{safe}.{suffix.lstrip('.')}"


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
