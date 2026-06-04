"""Proposal-only FX send and return recommendations.

This module is transport-free. It turns source role, tempo, arrangement, return
headroom, feedback risk, and intelligibility evidence into operator-facing FX
send proposals, but it never creates a live console write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


FX_METHOD_ID = "fx_send_proposal"
FX_VERSION = "fx_send_proposal_v1"
MIN_SEND_DB = -36.0
MAX_SEND_DB = -12.0
MAX_RETURN_LEVEL_DB = -3.0
MIN_RETURN_HEADROOM_DB = 3.0
HIGH_FEEDBACK_RISK = 0.55
HIGH_INTELLIGIBILITY_RISK = 0.65
MIN_SEND_DELTA_DB = 1.5

DRY_ANCHOR_FAMILIES = {"kick", "bass", "sub", "master", "reference"}
ROLE_TOKENS = (
    ("lead_vocal", ("lead_vocal", "lead vocal", "main_vocal", "main vocal", "vox lead")),
    ("backing_vocal", ("backing_vocal", "backing vocal", "back_vocal", "bgv", "bv")),
    ("kick", ("kick", "bd", "bass_drum")),
    ("snare", ("snare", "sn")),
    ("bass", ("bass", "di bass", "bass_di")),
    ("sub", ("sub", "subkick")),
    ("tom", ("tom", "floor_tom", "floor tom")),
    ("overhead", ("overhead", "oh", "cymbal")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("keys", "key", "piano", "synth", "organ")),
    ("percussion", ("perc", "clap", "shaker", "tamb")),
    ("playback", ("playback", "track", "stem", "loop")),
    ("master", ("master", "main", "reference")),
)


@dataclass(frozen=True)
class FXPreset:
    fx_type: str
    bus_name: str
    target_send_db: float
    return_level_db: float
    hpf_hz: float
    lpf_hz: float
    predelay_ms: float = 0.0
    decay_s: float = 0.0
    delay_note: str = ""
    ducking_db: float = 0.0


ROLE_PRESETS: Dict[str, Tuple[FXPreset, ...]] = {
    "lead_vocal": (
        FXPreset("reverb", "Vocal Plate", -18.0, -5.0, 220.0, 7800.0, predelay_ms=42.0, decay_s=1.55, ducking_db=1.8),
        FXPreset("delay", "Tempo Delay", -24.0, -5.0, 260.0, 5200.0, delay_note="dotted_eighth", ducking_db=4.0),
    ),
    "backing_vocal": (
        FXPreset("reverb", "Vocal Plate", -15.0, -6.0, 220.0, 7600.0, predelay_ms=36.0, decay_s=1.45, ducking_db=1.0),
        FXPreset("chorus", "Mod Doubler", -23.0, -11.0, 220.0, 8500.0),
    ),
    "snare": (
        FXPreset("reverb", "Drum Room", -17.0, -5.5, 180.0, 5600.0, predelay_ms=10.0, decay_s=0.72),
    ),
    "tom": (
        FXPreset("reverb", "Drum Room", -19.0, -5.5, 160.0, 5400.0, predelay_ms=8.0, decay_s=0.78),
    ),
    "overhead": (
        FXPreset("reverb", "Drum Room", -30.0, -8.0, 240.0, 5400.0, predelay_ms=4.0, decay_s=0.55),
    ),
    "guitar": (
        FXPreset("reverb", "Shared Plate", -25.0, -7.0, 220.0, 7200.0, predelay_ms=24.0, decay_s=1.2),
        FXPreset("chorus", "Mod Doubler", -26.0, -11.0, 240.0, 8200.0),
    ),
    "keys": (
        FXPreset("reverb", "Shared Plate", -24.0, -7.0, 220.0, 7600.0, predelay_ms=28.0, decay_s=1.35),
    ),
    "percussion": (
        FXPreset("reverb", "Small Room", -24.0, -8.0, 220.0, 6000.0, predelay_ms=6.0, decay_s=0.65),
    ),
    "playback": (
        FXPreset("reverb", "Shared Plate", -31.0, -9.0, 260.0, 7000.0, predelay_ms=20.0, decay_s=1.0),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _clip01(value: Any) -> float:
    return _clip(_safe_float(value), 0.0, 1.0)


def classify_source_family(source_role: str) -> str:
    text = str(source_role or "").strip().lower().replace("-", "_")
    spaced = text.replace("_", " ")
    for family, tokens in ROLE_TOKENS:
        if any(token in text or token in spaced for token in tokens):
            return family
    return "default"


def tempo_delay_ms(tempo_bpm: Any, note: str = "dotted_eighth") -> float:
    bpm = _clip(_safe_float(tempo_bpm, 120.0), 40.0, 240.0)
    quarter = 60000.0 / bpm
    if note == "eighth":
        return round(quarter * 0.5, 1)
    if note == "quarter":
        return round(quarter, 1)
    return round(quarter * 0.75, 1)


@dataclass(frozen=True)
class FXSendAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    current_send_db: Optional[float] = None
    tempo_bpm: float = 120.0
    arrangement_role: str = ""
    vocal_priority: bool = True
    contribution_share: float = 0.0
    return_headroom_db: float = 6.0
    feedback_risk: float = 0.0
    intelligibility_risk: float = 0.0
    source_confidence: float = 0.75
    persistence_cycles: int = 1
    requested_fx_type: Optional[str] = None


@dataclass(frozen=True)
class FXSendRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    fx_type: str
    bus_name: str
    current_send_db: Optional[float]
    target_send_db: float
    delta_send_db: Optional[float]
    return_level_db: float
    hpf_hz: float
    lpf_hz: float
    predelay_ms: float
    decay_s: float
    delay_ms: Optional[float]
    ducking_db: float
    confidence: float
    severity: str
    reason: str
    blockers: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        return payload

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer style proposal without live-apply fields."""

        return {
            "id": self.id,
            "method_id": FX_METHOD_ID,
            "source": FX_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: FX send proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "fx_send",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:fx_send",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "fx_type": self.fx_type,
                "bus_name": self.bus_name,
                "current_send_db": self.current_send_db,
                "target_send_db": self.target_send_db,
                "delta_send_db": self.delta_send_db,
                "return_level_db": self.return_level_db,
                "hpf_hz": self.hpf_hz,
                "lpf_hz": self.lpf_hz,
                "predelay_ms": self.predelay_ms,
                "decay_s": self.decay_s,
                "delay_ms": self.delay_ms,
                "ducking_db": self.ducking_db,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _severity(delta: Optional[float], feedback_risk: float, intelligibility_risk: float) -> str:
    magnitude = abs(delta or 0.0)
    if feedback_risk >= 0.45 or intelligibility_risk >= 0.5 or magnitude >= 8.0:
        return "high"
    if magnitude >= 3.0:
        return "medium"
    return "low"


def _select_preset(analysis: FXSendAnalysisInput, family: str) -> Optional[FXPreset]:
    presets = ROLE_PRESETS.get(family)
    if not presets:
        return None
    requested = str(analysis.requested_fx_type or "").strip().lower()
    if requested:
        return next((preset for preset in presets if preset.fx_type == requested), None)
    if family == "lead_vocal" and "space" in str(analysis.arrangement_role).lower():
        return presets[0]
    return presets[0]


def analyze_fx_send_need(
    analysis: FXSendAnalysisInput,
    *,
    min_confidence: float = 0.45,
) -> FXSendRecommendation:
    family = classify_source_family(analysis.source_role)
    preset = _select_preset(analysis, family)
    current = None if analysis.current_send_db is None else round(_safe_float(analysis.current_send_db), 1)
    feedback_risk = _clip01(analysis.feedback_risk)
    intelligibility_risk = _clip01(analysis.intelligibility_risk)
    contribution = _clip01(analysis.contribution_share)
    source_confidence = _clip01(analysis.source_confidence)
    headroom = _safe_float(analysis.return_headroom_db)
    persistence = max(1, int(analysis.persistence_cycles or 1))

    blockers = []
    if family in DRY_ANCHOR_FAMILIES:
        blockers.append("dry_anchor_source")
    if preset is None:
        blockers.append("no_fx_preset_for_source")
        preset = FXPreset("reverb", "No FX", -36.0, -12.0, 250.0, 6500.0)
    if headroom < MIN_RETURN_HEADROOM_DB:
        blockers.append("insufficient_fx_return_headroom")
    if feedback_risk >= HIGH_FEEDBACK_RISK:
        blockers.append("feedback_risk_high")
    if preset.fx_type in {"delay", "reverb"} and intelligibility_risk >= HIGH_INTELLIGIBILITY_RISK:
        blockers.append("intelligibility_risk_high")

    target = _clip(preset.target_send_db, MIN_SEND_DB, MAX_SEND_DB)
    if intelligibility_risk >= 0.45:
        target = min(MIN_SEND_DB, target - 4.0)
    if feedback_risk >= 0.35:
        target = min(MIN_SEND_DB, target - 5.0)
    target = round(_clip(target, MIN_SEND_DB, MAX_SEND_DB), 1)
    delta = None if current is None else round(target - current, 1)
    if delta is not None and abs(delta) < MIN_SEND_DELTA_DB:
        blockers.append("send_change_not_needed")

    delay_ms = tempo_delay_ms(analysis.tempo_bpm, preset.delay_note) if preset.fx_type == "delay" else None
    confidence = _clip01(
        0.35 * source_confidence
        + 0.20 * min(persistence / 3.0, 1.0)
        + 0.15 * (1.0 - feedback_risk)
        + 0.15 * (1.0 - intelligibility_risk)
        + 0.10 * min(max(headroom, 0.0) / 8.0, 1.0)
        + 0.05 * contribution
    )

    evidence = {
        "source": FX_VERSION,
        "source_role": analysis.source_role,
        "source_family": family,
        "tempo_bpm": round(_clip(_safe_float(analysis.tempo_bpm, 120.0), 40.0, 240.0), 1),
        "arrangement_role": analysis.arrangement_role,
        "vocal_priority": bool(analysis.vocal_priority),
        "contribution_share": round(contribution, 3),
        "return_headroom_db": round(headroom, 2),
        "feedback_risk": round(feedback_risk, 3),
        "intelligibility_risk": round(intelligibility_risk, 3),
        "source_confidence": round(source_confidence, 3),
        "persistence_cycles": persistence,
        "transport_free": True,
        "post_fader_send_required_before_promotion": True,
        "return_filtering_required": True,
    }

    status = "proposal_only" if not blockers and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = f"{label} can use {preset.bus_name} {preset.fx_type} send at {target:.1f} dB."
    else:
        reason = f"{label} does not meet safe FX send proposal criteria."

    return FXSendRecommendation(
        id=f"{FX_METHOD_ID}:ch{int(analysis.channel_id)}:{preset.fx_type}",
        status=status,
        action_type="fx_send_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=family,
        fx_type=preset.fx_type,
        bus_name=preset.bus_name,
        current_send_db=current,
        target_send_db=target,
        delta_send_db=delta,
        return_level_db=round(_clip(preset.return_level_db, MIN_SEND_DB, MAX_RETURN_LEVEL_DB), 1),
        hpf_hz=round(preset.hpf_hz, 1),
        lpf_hz=round(preset.lpf_hz, 1),
        predelay_ms=round(preset.predelay_ms, 1),
        decay_s=round(preset.decay_s, 2),
        delay_ms=delay_ms,
        ducking_db=round(preset.ducking_db if analysis.vocal_priority else 0.0, 1),
        confidence=round(confidence, 3),
        severity=_severity(delta, feedback_risk, intelligibility_risk),
        reason=reason,
        blockers=tuple(sorted(set(blockers))),
        evidence=evidence,
    )


def build_fx_send_proposals(
    analyses: Iterable[FXSendAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_fx_send_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
