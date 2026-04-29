"""
Runtime state policy for AutoFOH action permissions.

This keeps concert/soundcheck permissions explicit without forcing the
existing engine into a full show-control architecture yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

from autofoh_models import RuntimeState


ACTION_FAMILY_GAIN = "gain"
ACTION_FAMILY_TONAL_EQ = "tonal_eq"
ACTION_FAMILY_HPF = "hpf"
ACTION_FAMILY_FADER = "fader"
ACTION_FAMILY_PAN = "pan"
ACTION_FAMILY_COMPRESSOR = "compressor"
ACTION_FAMILY_FX = "fx"
ACTION_FAMILY_FEEDBACK = "feedback"
ACTION_FAMILY_EMERGENCY_FADER = "emergency_fader"
ACTION_FAMILY_PHASE = "phase"


@dataclass(frozen=True)
class RuntimeStateRule:
    allowed_families: Set[str] = field(default_factory=set)
    forbidden_families: Set[str] = field(default_factory=set)
    required_evidence_duration_sec: float = 0.0
    confidence_threshold: float = 0.0
    priority_weight: float = 1.0

    def allows(self, action_family: str) -> bool:
        if action_family in self.forbidden_families:
            return False
        if not self.allowed_families:
            return False
        return action_family in self.allowed_families


DEFAULT_RUNTIME_STATE_RULES: Dict[RuntimeState, RuntimeStateRule] = {
    RuntimeState.IDLE: RuntimeStateRule(set(), set()),
    RuntimeState.PREFLIGHT: RuntimeStateRule(set(), set()),
    RuntimeState.SILENCE_CAPTURE: RuntimeStateRule(
        {ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FADER, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_PHASE},
        required_evidence_duration_sec=1.0,
    ),
    RuntimeState.LINE_CHECK: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FADER, ACTION_FAMILY_FX},
        required_evidence_duration_sec=1.0,
    ),
    RuntimeState.SOURCE_LEARNING: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=1.5,
    ),
    RuntimeState.STEM_LEARNING: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
    ),
    RuntimeState.FULL_BAND_LEARNING: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
    ),
    RuntimeState.SNAPSHOT_LOCK: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
    ),
    RuntimeState.PRE_SHOW_CHECK: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
    ),
    RuntimeState.LOAD_SONG_SNAPSHOT: RuntimeStateRule(
        {ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FADER, ACTION_FAMILY_COMPRESSOR},
    ),
    RuntimeState.SONG_START_STABILIZE: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_FADER, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FX, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_HPF},
        required_evidence_duration_sec=3.0,
    ),
    RuntimeState.VERSE: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.5,
    ),
    RuntimeState.CHORUS: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
    ),
    RuntimeState.SOLO: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=2.0,
        priority_weight=1.2,
    ),
    RuntimeState.SPEECH: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FADER, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_FX},
        required_evidence_duration_sec=1.5,
        priority_weight=1.3,
    ),
    RuntimeState.BETWEEN_SONGS: RuntimeStateRule(
        {ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FADER, ACTION_FAMILY_COMPRESSOR},
    ),
    RuntimeState.EMERGENCY_FEEDBACK: RuntimeStateRule(
        {ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER},
        set(),
        required_evidence_duration_sec=0.0,
        confidence_threshold=0.0,
        priority_weight=2.0,
    ),
    RuntimeState.EMERGENCY_SPL: RuntimeStateRule(
        {ACTION_FAMILY_EMERGENCY_FADER},
        set(),
        required_evidence_duration_sec=0.0,
        priority_weight=2.0,
    ),
    RuntimeState.EMERGENCY_SIGNAL_LOSS: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_EMERGENCY_FADER},
        {ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_FX},
        required_evidence_duration_sec=0.0,
        priority_weight=1.5,
    ),
    RuntimeState.MANUAL_LOCK: RuntimeStateRule(set(), {
        ACTION_FAMILY_GAIN,
        ACTION_FAMILY_HPF,
        ACTION_FAMILY_TONAL_EQ,
        ACTION_FAMILY_COMPRESSOR,
        ACTION_FAMILY_FADER,
        ACTION_FAMILY_PAN,
        ACTION_FAMILY_FX,
        ACTION_FAMILY_FEEDBACK,
        ACTION_FAMILY_EMERGENCY_FADER,
        ACTION_FAMILY_PHASE,
    }),
    RuntimeState.ROLLBACK: RuntimeStateRule(
        {ACTION_FAMILY_GAIN, ACTION_FAMILY_HPF, ACTION_FAMILY_TONAL_EQ, ACTION_FAMILY_COMPRESSOR, ACTION_FAMILY_FADER, ACTION_FAMILY_PAN, ACTION_FAMILY_FX, ACTION_FAMILY_FEEDBACK, ACTION_FAMILY_EMERGENCY_FADER, ACTION_FAMILY_PHASE},
        set(),
        required_evidence_duration_sec=0.0,
        priority_weight=1.4,
    ),
}


class RuntimeStatePolicy:
    def __init__(self, rules: Dict[RuntimeState, RuntimeStateRule] | None = None):
        self.rules = dict(rules or DEFAULT_RUNTIME_STATE_RULES)

    def rule_for(self, runtime_state: RuntimeState) -> RuntimeStateRule:
        return self.rules.get(runtime_state, RuntimeStateRule())

    def is_action_allowed(self, runtime_state: RuntimeState, action_family: str) -> bool:
        return self.rule_for(runtime_state).allows(action_family)

    def forbidden_families(self, runtime_state: RuntimeState) -> Iterable[str]:
        return self.rule_for(runtime_state).forbidden_families
