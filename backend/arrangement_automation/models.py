"""Shared models for arrangement-aware level automation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


ROLE_UNKNOWN = "unknown"


@dataclass
class ChannelRoleInfo:
    """Musical role inferred for a mixer channel."""

    channel_id: int
    channel_name: str
    role: str
    confidence: float
    group: str
    priority: int
    allowed_gain_range_db: Tuple[float, float]
    attack_time_sec: float
    release_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityState:
    """Current activity state for a channel."""

    channel_id: int
    active: bool
    activity_confidence: float
    rms_db: float
    peak_db: float
    activity_duration_sec: float
    noise_floor_db: float = -80.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DensityState:
    """Arrangement Density Index result."""

    density_index: float
    density_label: str
    active_channel_count: int
    midrange_congestion: float
    low_end_occupancy: float
    masking_pressure_on_lead: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SectionState:
    """Rule-based song section estimate."""

    section: str
    confidence: float
    previous_section: str
    time_in_section_sec: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaskingSource:
    """A channel that may mask the current lead source."""

    channel_id: int
    role: str
    masking_score: float
    suggested_action: str
    suggested_gain_reduction_db: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PriorityState:
    """Current musical priority model."""

    primary_source_role: str
    primary_channel_id: Optional[int]
    protected_roles: List[str]
    secondary_roles: List[str]
    roles_to_duck: List[str]
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlannedOffset:
    """Requested relative fader offset before safety limiting."""

    channel_id: int
    channel_name: str
    role: str
    requested_offset_db: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyResult:
    """Safety-limited automation output."""

    safe_offsets: Dict[int, float]
    blocked_offsets: Dict[int, Dict[str, Any]]
    reasons: Dict[int, List[str]]
    final_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
