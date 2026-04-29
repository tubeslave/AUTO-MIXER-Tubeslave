"""Typed data models shared by offline and backend mix-agent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, numpy scalars and paths into JSON-safe values."""
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
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
class AnalysisContext:
    """Input context for one mix analysis pass."""

    stems_path: str = ""
    mix_path: str = ""
    reference_path: str = ""
    genre: str = "neutral"
    target_platform: str = "streaming"
    sample_rate: int = 0
    mode: str = "offline"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class AudioStem:
    """Loaded stem metadata used by the offline decision loop."""

    name: str
    role: str
    path: str
    sample_rate: int
    duration_sec: float


@dataclass
class TrackMetrics:
    """Metric bundle for a stem or stereo mix."""

    name: str
    role: str = "mix"
    sample_rate: int = 0
    duration_sec: float = 0.0
    level: Dict[str, Any] = field(default_factory=dict)
    spectral: Dict[str, Any] = field(default_factory=dict)
    dynamics: Dict[str, Any] = field(default_factory=dict)
    stereo: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class MixAction:
    """Explainable recommendation or bounded operation."""

    id: str
    action_type: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    safe_range: Dict[str, Any] = field(default_factory=dict)
    mode: str = "recommend"
    reason: str = ""
    expected_improvement: str = ""
    risk: str = ""
    confidence: float = 0.5
    reversible: bool = True
    backend_mapping: Dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class RuleIssue:
    """One ranked issue produced by the rule engine."""

    id: str
    group: str
    name: str
    severity: str
    explanation: str
    evidence: List[str] = field(default_factory=list)
    suggested_action: str = ""
    expected_improvement: str = ""
    affected_tracks: List[str] = field(default_factory=list)
    metrics_before_after: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    do_not_apply_constraints: List[str] = field(default_factory=list)
    risk: str = ""
    actions: List[MixAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class ReferenceComparison:
    """Loudness-normalized comparison against a reference track."""

    enabled: bool = False
    spectral_distance: float = 0.0
    band_differences_db: Dict[str, float] = field(default_factory=dict)
    stereo_width_difference: float = 0.0
    dynamic_range_difference_db: float = 0.0
    loudness_matched: bool = False
    limitations: List[str] = field(default_factory=list)


@dataclass
class MixAnalysis:
    """Complete analysis snapshot for the current mix."""

    context: AnalysisContext
    mix: TrackMetrics
    stems: Dict[str, TrackMetrics] = field(default_factory=dict)
    masking_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stem_relationships: Dict[str, Any] = field(default_factory=dict)
    reference: ReferenceComparison = field(default_factory=ReferenceComparison)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class QualityDashboard:
    """Independent quality scores; the aggregate is intentionally cautious."""

    technical_health_score: float = 0.0
    gain_staging_score: float = 0.0
    balance_score: float = 0.0
    tonal_balance_score: float = 0.0
    masking_score: float = 0.0
    dynamics_score: float = 0.0
    stereo_mono_score: float = 0.0
    space_clarity_score: float = 0.0
    reference_match_score: float = 0.0
    translation_score: float = 0.0
    artifact_risk_score: float = 0.0
    overall_recommendation_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class SuggestionPlan:
    """Analysis, ranked issues and suggested actions for one pass."""

    analysis: MixAnalysis
    issues: List[RuleIssue] = field(default_factory=list)
    actions: List[MixAction] = field(default_factory=list)
    dashboard: QualityDashboard = field(default_factory=QualityDashboard)
    applied_actions: List[MixAction] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)


@dataclass
class BackendChannelSnapshot:
    """Minimal backend/live-console channel state accepted by the bridge."""

    channel_id: int
    name: str = ""
    role: str = "unknown"
    metrics: Dict[str, Any] = field(default_factory=dict)
    current_fader_db: Optional[float] = None
    current_pan: Optional[float] = None


@dataclass
class BackendBridgeResult:
    """Result of translating or applying mix-agent suggestions to a real mixer."""

    proposed: List[MixAction] = field(default_factory=list)
    translated: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    blocked: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "suggest"

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)
