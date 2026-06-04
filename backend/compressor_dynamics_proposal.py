"""Proposal-only compressor recommendations from channel dynamics evidence.

This module is intentionally transport-free. It converts measured dynamics
features into operator-facing compressor proposals, but it never creates a live
mixer write request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple


COMPRESSOR_METHOD_ID = "compressor_dynamics_proposal"
COMPRESSOR_VERSION = "compressor_dynamics_proposal_v1"
MIN_THRESHOLD_DB = -50.0
MAX_THRESHOLD_DB = -6.0
MIN_RATIO = 1.2
MAX_RATIO = 8.0
MIN_ATTACK_MS = 2.0
MAX_ATTACK_MS = 120.0
MIN_RELEASE_MS = 40.0
MAX_RELEASE_MS = 1200.0
MAX_MAKEUP_DB = 3.0


@dataclass(frozen=True)
class CompressorRolePreset:
    family: str
    target_crest_factor_db: float
    crest_factor_trigger_db: float
    lra_trigger_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    knee_db: float
    target_gain_reduction_db: float


ROLE_PRESETS: Dict[str, CompressorRolePreset] = {
    "vocal": CompressorRolePreset("vocal", 9.0, 10.0, 5.5, 3.0, 18.0, 140.0, 4.0, 3.0),
    "bass": CompressorRolePreset("bass", 6.0, 7.0, 4.5, 4.0, 35.0, 220.0, 3.0, 4.0),
    "drum": CompressorRolePreset("drum", 13.0, 14.0, 6.0, 4.0, 8.0, 100.0, 2.5, 4.0),
    "percussion": CompressorRolePreset("percussion", 15.0, 16.0, 6.0, 5.0, 5.0, 80.0, 2.0, 4.5),
    "guitar": CompressorRolePreset("guitar", 9.0, 10.0, 6.5, 2.4, 22.0, 160.0, 3.0, 2.5),
    "keys": CompressorRolePreset("keys", 8.0, 9.0, 6.0, 2.2, 28.0, 190.0, 3.0, 2.0),
    "playback": CompressorRolePreset("playback", 7.0, 9.0, 5.0, 1.8, 35.0, 260.0, 4.0, 1.5),
    "default": CompressorRolePreset("default", 10.0, 11.0, 6.0, 2.5, 24.0, 180.0, 3.0, 2.5),
}

ROLE_TOKENS = (
    ("vocal", ("vocal", "vox", "voice", "speech", "lead", "backing")),
    ("bass", ("bass", "sub")),
    ("percussion", ("perc", "clap", "shaker", "tamb")),
    ("drum", ("kick", "snare", "tom", "drum", "overhead", "oh", "hat")),
    ("guitar", ("gtr", "guitar")),
    ("keys", ("key", "keys", "piano", "synth", "organ")),
    ("playback", ("playback", "track", "stem", "loop")),
)


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


def estimate_crest_factor_db(peak_dbfs: Any, rms_dbfs: Any) -> float:
    peak = _safe_float(peak_dbfs, -120.0)
    rms = _safe_float(rms_dbfs, -120.0)
    if peak <= -110.0 or rms <= -110.0:
        return 0.0
    return round(max(0.0, peak - rms), 2)


def classify_source_family(source_role: str) -> str:
    text = str(source_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    for family, tokens in ROLE_TOKENS:
        if any(token in text for token in tokens):
            return family
    return "default"


@dataclass(frozen=True)
class CompressorAnalysisInput:
    channel_id: int
    source_role: str
    channel_label: str = ""
    crest_factor_db: Optional[float] = None
    lra_db: float = 0.0
    rms_dbfs: float = -30.0
    peak_dbfs: float = -12.0
    lufs_short: float = -24.0
    true_peak_headroom_db: float = 6.0
    existing_gain_reduction_db: float = 0.0
    contribution_share: float = 0.0
    persistence_cycles: int = 1
    transient_density: float = 0.0
    loudness_stability: float = 0.5


@dataclass(frozen=True)
class CompressorRecommendation:
    id: str
    status: str
    action_type: str
    channel_id: int
    channel_label: str
    source_role: str
    source_family: str
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    knee_db: float
    makeup_gain_db: float
    target_gain_reduction_db: float
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
            "method_id": COMPRESSOR_METHOD_ID,
            "source": COMPRESSOR_METHOD_ID,
            "title": f"{self.channel_label or f'Ch {self.channel_id}'}: compressor proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "compressor",
            "severity": self.severity,
            "confidence": self.confidence,
            "queue_action": "not_queueable_until_promoted",
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": [
                "unsupported_value_type:compressor",
                "not_in_supervised_fader_gain_executor",
                "hardware_readback_and_rollback_not_proven",
                *self.blockers,
            ],
            "proposed_action": {
                "action_type": self.action_type,
                "threshold_db": self.threshold_db,
                "ratio": self.ratio,
                "attack_ms": self.attack_ms,
                "release_ms": self.release_ms,
                "knee_db": self.knee_db,
                "makeup_gain_db": self.makeup_gain_db,
                "target_gain_reduction_db": self.target_gain_reduction_db,
            },
            "evidence": {
                **self.evidence,
                "live_write": False,
                "queue_mutation": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }


def _severity_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _preset_for_role(source_role: str) -> CompressorRolePreset:
    return ROLE_PRESETS[classify_source_family(source_role)]


def analyze_compressor_need(
    analysis: CompressorAnalysisInput,
    *,
    min_confidence: float = 0.42,
) -> CompressorRecommendation:
    preset = _preset_for_role(analysis.source_role)
    crest_factor = (
        max(0.0, _safe_float(analysis.crest_factor_db))
        if analysis.crest_factor_db is not None
        else estimate_crest_factor_db(analysis.peak_dbfs, analysis.rms_dbfs)
    )
    lra_db = max(0.0, _safe_float(analysis.lra_db))
    headroom = _safe_float(analysis.true_peak_headroom_db)
    existing_gr = max(0.0, _safe_float(analysis.existing_gain_reduction_db))
    contribution = _clip01(analysis.contribution_share)
    persistence = max(1, int(analysis.persistence_cycles or 1))
    transient_density = _clip01(analysis.transient_density)
    stability = _clip01(analysis.loudness_stability)
    peak_dbfs = _safe_float(analysis.peak_dbfs, -120.0)
    rms_dbfs = _safe_float(analysis.rms_dbfs, -120.0)

    cf_excess = max(0.0, crest_factor - preset.target_crest_factor_db)
    lra_excess = max(0.0, lra_db - preset.lra_trigger_db)
    instability = max(0.0, 0.65 - stability)
    score = _clip01(
        0.35 * _clip01(cf_excess / 9.0)
        + 0.30 * _clip01(lra_excess / 8.0)
        + 0.15 * _clip01(instability / 0.65)
        + 0.10 * contribution
        + 0.10 * min(persistence / 3.0, 1.0)
        + 0.05 * transient_density
    )
    needs_compression = crest_factor >= preset.crest_factor_trigger_db or lra_db >= preset.lra_trigger_db
    confidence = _clip01(score + (0.08 if needs_compression else 0.0))

    blockers = []
    if peak_dbfs < -55.0 or rms_dbfs < -70.0:
        blockers.append("signal_below_analysis_floor")
    if not needs_compression:
        blockers.append("compression_not_needed")
    if existing_gr >= 8.0:
        blockers.append("existing_compression_already_heavy")
    if headroom < 1.0:
        blockers.append("insufficient_true_peak_headroom_for_makeup")

    target_gr = _clip(preset.target_gain_reduction_db + score * 1.5, 1.0, 6.0)
    ratio = _clip(preset.ratio + score * 1.4, MIN_RATIO, MAX_RATIO)
    attack = _clip(preset.attack_ms * (0.85 if cf_excess > 4.0 else 1.0), MIN_ATTACK_MS, MAX_ATTACK_MS)
    release = _clip(preset.release_ms * (1.0 + instability * 0.35), MIN_RELEASE_MS, MAX_RELEASE_MS)
    threshold = _clip(rms_dbfs + 3.0 - score * 4.0, MIN_THRESHOLD_DB, MAX_THRESHOLD_DB)
    makeup_budget = max(0.0, headroom - 1.0)
    makeup = _clip(min(target_gr * 0.45, makeup_budget, MAX_MAKEUP_DB), 0.0, MAX_MAKEUP_DB)

    evidence = {
        "source": COMPRESSOR_VERSION,
        "source_role": analysis.source_role,
        "source_family": preset.family,
        "crest_factor_db": round(crest_factor, 2),
        "lra_db": round(lra_db, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "peak_dbfs": round(peak_dbfs, 2),
        "lufs_short": round(_safe_float(analysis.lufs_short, -120.0), 2),
        "true_peak_headroom_db": round(headroom, 2),
        "existing_gain_reduction_db": round(existing_gr, 2),
        "contribution_share": round(contribution, 3),
        "persistence_cycles": persistence,
        "transient_density": round(transient_density, 3),
        "loudness_stability": round(stability, 3),
        "target_crest_factor_db": preset.target_crest_factor_db,
        "crest_factor_trigger_db": preset.crest_factor_trigger_db,
        "lra_trigger_db": preset.lra_trigger_db,
        "transport_free": True,
        "makeup_gain_is_bounded_by_headroom": True,
    }

    status = "proposal_only" if not blockers and confidence >= min_confidence else "analysis_only"
    label = analysis.channel_label or f"Ch {analysis.channel_id}"
    if status == "proposal_only":
        reason = (
            f"{label} has uneven dynamics "
            f"(CF {crest_factor:.1f} dB, LRA {lra_db:.1f} dB)."
        )
    else:
        reason = f"{label} does not meet safe compressor proposal criteria."

    return CompressorRecommendation(
        id=f"{COMPRESSOR_METHOD_ID}:ch{int(analysis.channel_id)}",
        status=status,
        action_type="compressor_proposal",
        channel_id=int(analysis.channel_id),
        channel_label=label,
        source_role=str(analysis.source_role or "unknown"),
        source_family=preset.family,
        threshold_db=round(threshold, 1),
        ratio=round(ratio, 2),
        attack_ms=round(attack, 1),
        release_ms=round(release, 1),
        knee_db=round(preset.knee_db, 1),
        makeup_gain_db=round(makeup, 1),
        target_gain_reduction_db=round(target_gr, 1),
        confidence=round(confidence, 3),
        severity=_severity_from_score(score),
        reason=reason,
        blockers=tuple(sorted(set(blockers))),
        evidence=evidence,
    )


def build_compressor_proposals(
    analyses: Iterable[CompressorAnalysisInput],
) -> Tuple[Dict[str, Any], ...]:
    proposals = []
    for analysis in analyses:
        recommendation = analyze_compressor_need(analysis)
        if recommendation.status == "proposal_only":
            proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)
