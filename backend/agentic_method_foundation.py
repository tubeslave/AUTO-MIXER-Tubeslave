"""Dry-run foundation layer for agentic live-mixing recommendations.

This module is intentionally transport-free: it extracts features and produces
operator-facing recommendations, but it never imports mixer clients and never
sends OSC/MIDI/live writes. Promotion to a real console action must happen
through the existing supervised Product Layer gate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np

try:
    from .context_aware_gate import BleedTelemetry, ContextAwareGateAgent
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from context_aware_gate import BleedTelemetry, ContextAwareGateAgent


FOUNDATION_VERSION = "agentic_foundation_v1"
EPSILON = 1e-12
DEFAULT_SAMPLE_RATE = 48000

FREQUENCY_BANDS = {
    "sub": (20.0, 60.0),
    "low": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 2000.0),
    "presence": (2000.0, 6000.0),
    "air": (6000.0, 16000.0),
}

LIVE_WRITE_BLOCKED_ACTIONS = {
    "eq",
    "compressor",
    "gate",
    "expander",
    "pan",
    "fx_send",
    "routing",
    "snapshot",
    "training_rule",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _round(value: Any, digits: int = 2) -> Optional[float]:
    numeric = _safe_float(value)
    return round(numeric, digits) if numeric is not None else None


def _dbfs(amplitude: float) -> float:
    if amplitude <= EPSILON:
        return -120.0
    return 20.0 * math.log10(max(float(amplitude), EPSILON))


def _amplitude_from_db(db_value: Optional[float]) -> float:
    if db_value is None:
        return 0.0
    return float(10 ** (float(db_value) / 20.0))


def _power_sum_db(values: Iterable[Optional[float]]) -> Optional[float]:
    powers = [10 ** (float(value) / 10.0) for value in values if _safe_float(value) is not None]
    if not powers:
        return None
    return 10.0 * math.log10(max(sum(powers), EPSILON))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _frame_rms_db(samples: np.ndarray, frame_size: int = 1024) -> List[float]:
    if samples.size == 0:
        return []
    if samples.size < frame_size:
        return [_dbfs(float(np.sqrt(np.mean(samples ** 2))))] if samples.size else []
    frame_count = samples.size // frame_size
    trimmed = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    return [_dbfs(float(value)) for value in rms]


@dataclass(frozen=True)
class AgenticFoundationConfig:
    """Safe defaults for recommendation-only agentic methods."""

    live_writes_enabled: bool = False
    dry_run: bool = True
    max_gain_change_db_per_step: float = 0.5
    max_fader_move_db_per_step: float = 1.0
    max_eq_boost_db: float = 3.0
    max_eq_cut_db: float = 4.0
    confidence_threshold: float = 0.7
    rollback_required: bool = True
    peak_watch_db: float = -3.0
    clip_risk_db: float = -1.0
    target_peak_db: float = -6.0
    low_activity_threshold: float = 0.08

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChannelFeatureSnapshot:
    channel_id: int
    role: str = "unknown"
    group: str = "unknown"
    rms_db: Optional[float] = None
    peak_db: Optional[float] = None
    true_peak_db: Optional[float] = None
    lufs: Optional[float] = None
    crest_factor_db: Optional[float] = None
    headroom_db: Optional[float] = None
    clipping_count: int = 0
    noise_floor_db: Optional[float] = None
    signal_activity: float = 0.0
    bleed_index: Optional[float] = None
    gain_change_slew_rate_db_per_sec: Optional[float] = None
    spectral_centroid_hz: Optional[float] = None
    spectral_tilt_db_per_oct: Optional[float] = None
    band_energy_db: Dict[str, Optional[float]] = field(default_factory=dict)
    resonance_score: Optional[float] = None
    harshness_score: Optional[float] = None
    dynamic_range_db: Optional[float] = None
    suggested_gain_reduction_range_db: Optional[List[float]] = None
    transient_preservation_estimate: Optional[float] = None
    pumping_risk_score: Optional[float] = None
    false_open_risk: Optional[float] = None
    false_close_risk: Optional[float] = None
    chatter_risk: Optional[float] = None
    current_gain_db: Optional[float] = None
    current_fader_db: Optional[float] = None
    timestamp: str = field(default_factory=utc_now)
    source: str = FOUNDATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterChannelContext:
    active_channel_count: int = 0
    vocal_to_band_ratio_db: Optional[float] = None
    kick_bass_relation_db: Optional[float] = None
    master_bus_spectral_balance: Dict[str, Optional[float]] = field(default_factory=dict)
    mono_compatibility_risk: Optional[float] = None
    masking_index: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRecommendation:
    channel_id: int
    agent_name: str
    action_type: str
    suggested_value: Optional[float] = None
    suggested_params: Dict[str, Any] = field(default_factory=dict)
    current_value: Optional[float] = None
    confidence: float = 0.0
    risk_score: float = 1.0
    reason: str = ""
    safety_status: str = "dry_run_only"
    dry_run: bool = True
    timestamp: str = field(default_factory=utc_now)
    idempotency_key: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if payload["idempotency_key"] is None:
            payload["idempotency_key"] = recommendation_idempotency_key(payload)
        return payload


@dataclass(frozen=True)
class OfflineTeacherExample:
    example_id: str
    run_id: str
    channel_id: int
    feature_snapshot: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    reference_metrics: Dict[str, Any] = field(default_factory=dict)
    operator_decision: Optional[str] = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecommendationModel(Protocol):
    def recommend(
        self,
        snapshot: ChannelFeatureSnapshot,
        context: Optional[InterChannelContext] = None,
    ) -> List[AgentRecommendation]:
        ...


def recommendation_idempotency_key(payload: Dict[str, Any]) -> str:
    stable = "|".join(
        str(payload.get(key))
        for key in (
            "channel_id",
            "agent_name",
            "action_type",
            "current_value",
            "suggested_value",
            "safety_status",
        )
    )
    return f"rec-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}"


def _make_recommendation(**kwargs: Any) -> AgentRecommendation:
    rec = AgentRecommendation(**kwargs)
    payload = rec.to_dict()
    return AgentRecommendation(**payload)


def extract_channel_features(
    samples: Iterable[float] | np.ndarray,
    *,
    channel_id: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    role: str = "unknown",
    group: str = "unknown",
    peer_samples: Optional[Iterable[Iterable[float] | np.ndarray]] = None,
    current_gain_db: Optional[float] = None,
    current_fader_db: Optional[float] = None,
) -> ChannelFeatureSnapshot:
    """Extract level, spectral, dynamics and bleed features from one channel.

    True peak is approximated by sample peak here; the existing
    ``signal_metrics.SignalAnalyzer`` remains the richer online analyzer when
    the full audio service is available.
    """

    data = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, dtype=float).reshape(-1)
    if data.size == 0:
        return ChannelFeatureSnapshot(channel_id=channel_id, role=role, group=group)

    data = np.nan_to_num(data, copy=False)
    peak_amp = float(np.max(np.abs(data)))
    rms_amp = float(np.sqrt(np.mean(data ** 2)))
    peak_db = _dbfs(peak_amp)
    rms_db = _dbfs(rms_amp)
    true_peak_db = peak_db
    crest = max(0.0, true_peak_db - rms_db)
    frame_levels = _frame_rms_db(data)
    active_frames = [level for level in frame_levels if level > -80.0]
    noise_floor = float(np.percentile(frame_levels, 10)) if frame_levels else -120.0
    active_threshold = min(noise_floor + 6.0, -50.0)
    activity = (
        sum(1 for level in frame_levels if level >= active_threshold) / len(frame_levels)
        if frame_levels
        else float(np.mean(np.abs(data) > 0.001))
    )
    dynamic_range = (
        float(np.percentile(active_frames, 95) - np.percentile(active_frames, 10))
        if len(active_frames) >= 3
        else 0.0
    )
    clipping_count = int(np.sum(np.abs(data) >= 0.999))

    band_energy, centroid_hz, tilt_db_per_oct, resonance_score, harshness_score = _spectral_features(
        data,
        sample_rate=sample_rate,
    )
    bleed_index = _bleed_index(data, peer_samples)
    pumping_risk = _clip01((8.0 - dynamic_range) / 8.0) if dynamic_range is not None else None
    chatter = _chatter_risk(frame_levels, active_threshold)

    return ChannelFeatureSnapshot(
        channel_id=int(channel_id),
        role=role or "unknown",
        group=group or "unknown",
        rms_db=_round(rms_db),
        peak_db=_round(peak_db),
        true_peak_db=_round(true_peak_db),
        lufs=_round(rms_db - 1.5),
        crest_factor_db=_round(crest),
        headroom_db=_round(-true_peak_db),
        clipping_count=clipping_count,
        noise_floor_db=_round(noise_floor),
        signal_activity=_round(_clip01(activity), 3) or 0.0,
        bleed_index=_round(bleed_index, 3),
        spectral_centroid_hz=_round(centroid_hz, 1),
        spectral_tilt_db_per_oct=_round(tilt_db_per_oct, 2),
        band_energy_db={key: _round(value) for key, value in band_energy.items()},
        resonance_score=_round(resonance_score, 3),
        harshness_score=_round(harshness_score, 3),
        dynamic_range_db=_round(dynamic_range),
        suggested_gain_reduction_range_db=[1.0, 4.0] if crest >= 12.0 else [0.0, 2.0],
        transient_preservation_estimate=_round(_clip01(crest / 18.0), 3),
        pumping_risk_score=_round(pumping_risk, 3),
        false_open_risk=_round(_clip01((bleed_index or 0.0) * (1.0 - activity)), 3),
        false_close_risk=_round(_clip01((1.0 - activity) if rms_db > -45.0 else 0.0), 3),
        chatter_risk=_round(chatter, 3),
        current_gain_db=_round(current_gain_db),
        current_fader_db=_round(current_fader_db),
    )


def _spectral_features(
    data: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[Dict[str, float], float, float, float, float]:
    if data.size < 32:
        return {band: -120.0 for band in FREQUENCY_BANDS}, 0.0, 0.0, 0.0, 0.0
    fft_size = min(max(2048, int(2 ** math.ceil(math.log2(min(data.size, 8192))))), data.size)
    window = np.hanning(fft_size)
    frame = data[:fft_size] * window
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    total = float(np.sum(spectrum) + EPSILON)
    centroid = float(np.sum(freqs * spectrum) / total)
    band_energy: Dict[str, float] = {}
    for band, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        energy = float(np.sqrt(np.mean(spectrum[mask] ** 2))) if np.any(mask) else 0.0
        band_energy[band] = _dbfs(energy / max(fft_size, 1))

    valid = (freqs >= 80.0) & (freqs <= min(16000.0, sample_rate / 2.0)) & (spectrum > EPSILON)
    if np.count_nonzero(valid) >= 3:
        x = np.log2(freqs[valid] / 1000.0)
        y = 20.0 * np.log10(spectrum[valid] + EPSILON)
        tilt = float(np.polyfit(x, y, 1)[0])
        resonance = float(np.max(y) - np.median(y))
    else:
        tilt = 0.0
        resonance = 0.0
    harsh_energy = _amplitude_from_db(band_energy.get("presence")) + _amplitude_from_db(band_energy.get("air"))
    low_mid_energy = _amplitude_from_db(band_energy.get("low_mid")) + _amplitude_from_db(band_energy.get("mid"))
    harshness = harsh_energy / max(harsh_energy + low_mid_energy, EPSILON)
    return band_energy, centroid, tilt, _clip01(resonance / 24.0), _clip01(harshness)


def _bleed_index(
    samples: np.ndarray,
    peer_samples: Optional[Iterable[Iterable[float] | np.ndarray]],
) -> Optional[float]:
    if not peer_samples:
        return None
    scores: List[float] = []
    for peer in peer_samples:
        peer_data = np.asarray(list(peer) if not isinstance(peer, np.ndarray) else peer, dtype=float).reshape(-1)
        size = min(samples.size, peer_data.size)
        if size < 16:
            continue
        a = samples[:size]
        b = peer_data[:size]
        if np.std(a) <= EPSILON or np.std(b) <= EPSILON:
            continue
        scores.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return max(scores) if scores else None


def _chatter_risk(frame_levels: List[float], active_threshold: float) -> float:
    if len(frame_levels) < 4:
        return 0.0
    states = [level >= active_threshold for level in frame_levels]
    transitions = sum(1 for previous, current in zip(states, states[1:]) if previous != current)
    return _clip01(transitions / max(len(states) - 1, 1))


def feature_snapshot_from_channel_inventory(channel: Dict[str, Any]) -> ChannelFeatureSnapshot:
    meter = channel.get("meter", {}) if isinstance(channel.get("meter"), dict) else {}
    peak = _safe_float(channel.get("peak_db") if channel.get("peak_db") is not None else meter.get("peak_db"))
    rms = _safe_float(channel.get("level_db") if channel.get("level_db") is not None else meter.get("rms_db"))
    lufs = _safe_float(channel.get("lufs") if channel.get("lufs") is not None else meter.get("lufs"))
    dynamic = _safe_float(channel.get("dynamic_range_db") if channel.get("dynamic_range_db") is not None else meter.get("dynamic_range_db"))
    control = channel.get("control_state", {}) if isinstance(channel.get("control_state"), dict) else {}
    crest = max(0.0, peak - rms) if peak is not None and rms is not None else None
    return ChannelFeatureSnapshot(
        channel_id=int(channel.get("id") or channel.get("number") or 0),
        role=str(channel.get("role") or "unknown"),
        group=str(channel.get("group") or "unknown"),
        rms_db=_round(rms),
        peak_db=_round(peak),
        true_peak_db=_round(peak),
        lufs=_round(lufs),
        crest_factor_db=_round(crest),
        headroom_db=_round(-peak) if peak is not None else None,
        clipping_count=1 if peak is not None and peak >= -0.1 else 0,
        signal_activity=1.0 if rms is not None and rms > -80.0 else 0.0,
        dynamic_range_db=_round(dynamic if dynamic is not None else crest),
        suggested_gain_reduction_range_db=[1.0, 4.0] if crest is not None and crest >= 12.0 else [0.0, 2.0],
        current_gain_db=_round(control.get("gain_db")),
        current_fader_db=_round(control.get("fader_db")),
    )


def _snapshot_has_any_role_or_group(snapshot: ChannelFeatureSnapshot, *tokens: str) -> bool:
    text = f"{snapshot.role} {snapshot.group}".lower()
    return any(token in text for token in tokens)


def build_inter_channel_context(snapshots: Iterable[ChannelFeatureSnapshot]) -> InterChannelContext:
    rows = list(snapshots)
    active = [row for row in rows if row.signal_activity > 0.0]
    vocal_lufs = [
        row.lufs
        for row in active
        if row.lufs is not None and _snapshot_has_any_role_or_group(row, "vocal")
    ]
    band_lufs = [
        row.lufs
        for row in active
        if row.lufs is not None and not _snapshot_has_any_role_or_group(row, "vocal")
    ]
    vocal_sum = _power_sum_db(vocal_lufs)
    band_sum = _power_sum_db(band_lufs)
    kick = _power_sum_db(row.lufs for row in active if "kick" in row.role.lower())
    bass = _power_sum_db(row.lufs for row in active if "bass" in row.role.lower())

    master_bands: Dict[str, Optional[float]] = {}
    for band in FREQUENCY_BANDS:
        master_bands[band] = _power_sum_db(row.band_energy_db.get(band) for row in active)

    masking: Dict[str, float] = {}
    vocals = [row for row in active if _snapshot_has_any_role_or_group(row, "vocal")]
    guitars = [
        row
        for row in active
        if _snapshot_has_any_role_or_group(row, "guitar", "keys")
    ]
    if vocals and guitars:
        vocal_presence = _power_sum_db(row.band_energy_db.get("presence") for row in vocals)
        guitar_presence = _power_sum_db(row.band_energy_db.get("presence") for row in guitars)
        if vocal_presence is not None and guitar_presence is not None:
            masking["vocal_presence_vs_guitar_keys"] = _clip01((guitar_presence - vocal_presence + 12.0) / 24.0)

    return InterChannelContext(
        active_channel_count=len(active),
        vocal_to_band_ratio_db=_round(vocal_sum - band_sum) if vocal_sum is not None and band_sum is not None else None,
        kick_bass_relation_db=_round(kick - bass) if kick is not None and bass is not None else None,
        master_bus_spectral_balance=master_bands,
        mono_compatibility_risk=None,
        masking_index=masking,
    )


class BaselineAgenticRecommender:
    """Small replaceable heuristic model for dry-run recommendations."""

    def __init__(self, config: Optional[AgenticFoundationConfig] = None):
        self.config = config or AgenticFoundationConfig()

    def recommend(
        self,
        snapshot: ChannelFeatureSnapshot,
        context: Optional[InterChannelContext] = None,
    ) -> List[AgentRecommendation]:
        recommendations: List[AgentRecommendation] = []
        recommendations.extend(_gain_balance_recommendations(snapshot, context, self.config))
        recommendations.extend(_eq_recommendations(snapshot, context, self.config))
        recommendations.extend(_dynamics_recommendations(snapshot, self.config))
        recommendations.extend(_gate_recommendations(snapshot, self.config))
        recommendations.extend(_pan_recommendations(snapshot, self.config))
        recommendations.extend(_fx_recommendations(snapshot, self.config))
        return recommendations


def build_agentic_recommendations(
    snapshot: ChannelFeatureSnapshot,
    context: Optional[InterChannelContext] = None,
    config: Optional[AgenticFoundationConfig] = None,
) -> List[AgentRecommendation]:
    return BaselineAgenticRecommender(config).recommend(snapshot, context)


def _gain_balance_recommendations(
    snapshot: ChannelFeatureSnapshot,
    context: Optional[InterChannelContext],
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    recs: List[AgentRecommendation] = []
    peak = snapshot.peak_db
    lufs = snapshot.lufs
    current_gain = snapshot.current_gain_db
    current_fader = snapshot.current_fader_db
    if peak is not None and (snapshot.clipping_count > 0 or peak >= config.clip_risk_db):
        reduction = min(config.max_gain_change_db_per_step, max(0.1, peak - config.target_peak_db))
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="Auto Gain / Balance Agent",
                action_type="gain_trim",
                suggested_value=_round((current_gain if current_gain is not None else 0.0) - reduction, 1),
                current_value=current_gain,
                confidence=0.9,
                risk_score=0.18,
                reason="Input peak is near clipping; recommend a small trim-down candidate for supervised review.",
                safety_status="dry_run_only",
                dry_run=True,
                evidence={
                    "peak_db": peak,
                    "target_peak_db": config.target_peak_db,
                    "max_step_db": config.max_gain_change_db_per_step,
                    "rollback_required": config.rollback_required,
                },
            )
        )
    elif peak is not None and peak >= config.peak_watch_db:
        reduction = min(config.max_fader_move_db_per_step, max(0.1, peak - config.target_peak_db))
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="Auto Gain / Balance Agent",
                action_type="fader",
                suggested_value=_round((current_fader if current_fader is not None else 0.0) - reduction, 1),
                current_value=current_fader,
                confidence=0.84,
                risk_score=0.24,
                reason="Channel peak is close to the safety watch zone; recommend a small fader-down candidate.",
                safety_status="dry_run_only",
                dry_run=True,
                evidence={"peak_db": peak, "max_step_db": config.max_fader_move_db_per_step},
            )
        )
    elif (
        lufs is not None
        and peak is not None
        and lufs < -34.0
        and peak < -12.0
        and snapshot.signal_activity >= config.low_activity_threshold
    ):
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="Auto Gain / Balance Agent",
                action_type="gain_review",
                suggested_value=config.max_gain_change_db_per_step,
                current_value=current_gain,
                confidence=0.62,
                risk_score=0.55,
                reason="Signal is quiet but upward gain is audit-only until the operator authors or approves a supervised proposal.",
                safety_status="blocked",
                dry_run=True,
                evidence={"lufs": lufs, "peak_db": peak, "blocked_reason": "auto_generated_upward_gain_blocked"},
            )
        )

    if context and context.vocal_to_band_ratio_db is not None and "vocal" in snapshot.role.lower():
        ratio = context.vocal_to_band_ratio_db
        if ratio < -8.0:
            recs.append(
                _make_recommendation(
                    channel_id=snapshot.channel_id,
                    agent_name="Balance Agent",
                    action_type="balance_review",
                    suggested_params={"target": "vocal_to_band_ratio", "direction": "raise_vocal_or_lower_band"},
                    confidence=0.68,
                    risk_score=0.42,
                    reason="Vocal appears buried against the band; keep as review evidence, not an automatic gain-up.",
                    safety_status="blocked",
                    dry_run=True,
                    evidence={"vocal_to_band_ratio_db": ratio},
                )
            )
    return recs


def _eq_recommendations(
    snapshot: ChannelFeatureSnapshot,
    context: Optional[InterChannelContext],
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    recs: List[AgentRecommendation] = []
    role = snapshot.role.lower()
    harshness = snapshot.harshness_score
    low_mid = snapshot.band_energy_db.get("low_mid")
    mid = snapshot.band_energy_db.get("mid")
    if harshness is not None and harshness >= 0.62:
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="AutoEQ Agent",
                action_type="eq",
                suggested_params={
                    "band": "presence",
                    "frequency_hz": 3500 if "vocal" in role else 4200,
                    "gain_db": -min(config.max_eq_cut_db, 2.5),
                    "q": 1.4,
                },
                confidence=0.72,
                risk_score=0.38,
                reason="Presence/air energy is high; suggest a bounded harshness-control cut for offline or operator review.",
                safety_status="proposal_only",
                dry_run=True,
                evidence={"harshness_score": harshness, "max_eq_boost_db": config.max_eq_boost_db},
            )
        )
    if "vocal" in role and low_mid is not None and mid is not None and low_mid > mid - 2.0:
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="AutoEQ Agent",
                action_type="eq",
                suggested_params={"band": "low_mid", "frequency_hz": 280, "gain_db": -2.0, "q": 1.0},
                confidence=0.66,
                risk_score=0.32,
                reason="Vocal low-mid energy may be masking clarity; recommend a small mud-control cut.",
                safety_status="proposal_only",
                dry_run=True,
                evidence={"low_mid_db": low_mid, "mid_db": mid},
            )
        )
    if context and context.masking_index.get("vocal_presence_vs_guitar_keys", 0.0) >= 0.65 and "guitar" in role:
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="AutoEQ Agent",
                action_type="eq",
                suggested_params={"band": "presence", "frequency_hz": 3000, "gain_db": -1.5, "q": 1.2},
                confidence=0.64,
                risk_score=0.35,
                reason="Guitar/key presence may mask vocal intelligibility; keep as masking-aware EQ proposal.",
                safety_status="proposal_only",
                dry_run=True,
                evidence={"masking_index": context.masking_index},
            )
        )
    return recs


def _dynamics_recommendations(
    snapshot: ChannelFeatureSnapshot,
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    recs: List[AgentRecommendation] = []
    dynamic = snapshot.dynamic_range_db
    crest = snapshot.crest_factor_db
    if dynamic is not None and crest is not None and (dynamic >= 14.0 or crest >= 14.0):
        recs.append(
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="Compression Agent",
                action_type="compressor",
                suggested_params={
                    "threshold_db": _round((snapshot.rms_db or -24.0) + 3.0, 1),
                    "ratio": 2.0 if "vocal" in snapshot.role.lower() else 3.0,
                    "attack_ms": 15.0,
                    "release_ms": 120.0,
                    "target_gain_reduction_db": [2.0, 4.0],
                },
                confidence=0.7,
                risk_score=0.44,
                reason="Dynamic range/crest factor is high; suggest conservative compression settings for review only.",
                safety_status="proposal_only",
                dry_run=True,
                evidence={
                    "dynamic_range_db": dynamic,
                    "crest_factor_db": crest,
                    "pumping_risk_score": snapshot.pumping_risk_score,
                },
            )
        )
    return recs


def _gate_recommendations(
    snapshot: ChannelFeatureSnapshot,
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    _ = config
    recs: List[AgentRecommendation] = []
    telemetry = BleedTelemetry.from_feature_snapshot(snapshot, channel_label=snapshot.role)
    context_recommendation = ContextAwareGateAgent().recommend(telemetry)
    if context_recommendation is not None:
        payload = context_recommendation.to_dict()
        recs.append(
            _make_recommendation(
                channel_id=int(payload["channel_id"]),
                agent_name=str(payload["agent_name"]),
                action_type=str(payload["action_type"]),
                suggested_params=dict(payload.get("suggested_params") or {}),
                confidence=float(payload.get("confidence") or 0.0),
                risk_score=float(payload.get("risk_score") or 1.0),
                reason=str(payload.get("reason") or "Context-aware gate/expander review."),
                safety_status=str(payload.get("safety_status") or "proposal_only"),
                dry_run=True,
                idempotency_key=payload.get("idempotency_key"),
                evidence=dict(payload.get("evidence") or {}),
            )
        )
    return recs


def _pan_recommendations(
    snapshot: ChannelFeatureSnapshot,
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    _ = config
    role = snapshot.role.lower()
    if any(key in role for key in ("kick", "bass", "vocal", "snare")):
        return [
            _make_recommendation(
                channel_id=snapshot.channel_id,
                agent_name="Panning Agent",
                action_type="pan",
                suggested_value=0.0,
                confidence=0.58,
                risk_score=0.18,
                reason="Core source should stay near center for mono compatibility; this is a review-only pan prior.",
                safety_status="proposal_only",
                dry_run=True,
                evidence={"mono_compatibility": "center_priority", "stage_position_source": "not_available"},
            )
        ]
    return []


def _fx_recommendations(
    snapshot: ChannelFeatureSnapshot,
    config: AgenticFoundationConfig,
) -> List[AgentRecommendation]:
    _ = config
    if "vocal" not in snapshot.role.lower():
        return []
    return [
        _make_recommendation(
            channel_id=snapshot.channel_id,
            agent_name="FX / Color Agent",
            action_type="fx_send",
            suggested_params={"send_type": "vocal_reverb", "send_db_range": [-30.0, -18.0], "dry_wet_limit": 0.18},
            confidence=0.52,
            risk_score=0.5,
            reason="Vocal FX prior is available for operator taste; no live send write is allowed by this layer.",
            safety_status="proposal_only",
            dry_run=True,
            evidence={"tempo_available": False, "live_write_blocked": True},
        )
    ]


def safety_review_recommendations(
    recommendations: Iterable[AgentRecommendation],
    *,
    config: Optional[AgenticFoundationConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AgenticFoundationConfig()
    rows = [rec.to_dict() for rec in recommendations]
    blocked = []
    candidates = []
    for row in rows:
        action_type = str(row.get("action_type") or "")
        confidence = _safe_float(row.get("confidence")) or 0.0
        reasons: List[str] = []
        if action_type in LIVE_WRITE_BLOCKED_ACTIONS:
            reasons.append(f"{action_type}_live_write_quarantined")
        if confidence < cfg.confidence_threshold:
            reasons.append("confidence_below_threshold")
        if not cfg.live_writes_enabled or cfg.dry_run:
            reasons.append("dry_run_policy")
        if reasons:
            row["safety_status"] = "blocked" if action_type in {"routing", "snapshot", "training_rule"} else row.get("safety_status", "dry_run_only")
            row["blocked_reasons"] = reasons
            blocked.append(row)
        else:
            candidates.append(row)
    return {
        "type": "agentic_recommendation_safety_review",
        "generated_at": utc_now(),
        "policy": cfg.to_dict(),
        "summary": {
            "recommendation_count": len(rows),
            "blocked_count": len(blocked),
            "supervised_candidate_count": len(candidates),
            "live_writes_enabled": cfg.live_writes_enabled,
            "dry_run": cfg.dry_run,
        },
        "blocked": blocked,
        "supervised_candidates": candidates,
    }


def build_offline_teacher_example(
    *,
    run_id: str,
    snapshot: ChannelFeatureSnapshot,
    recommendations: Iterable[AgentRecommendation],
    reference_metrics: Optional[Dict[str, Any]] = None,
    operator_decision: Optional[str] = None,
) -> OfflineTeacherExample:
    payload = snapshot.to_dict()
    recs = [rec.to_dict() for rec in recommendations]
    stable = f"{run_id}|{snapshot.channel_id}|{len(recs)}|{snapshot.timestamp}"
    return OfflineTeacherExample(
        example_id=f"teacher-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}",
        run_id=run_id,
        channel_id=snapshot.channel_id,
        feature_snapshot=payload,
        recommendations=recs,
        reference_metrics=reference_metrics or {},
        operator_decision=operator_decision,
    )


def build_offline_teacher_examples_from_channel_inventory(
    channels: Iterable[Dict[str, Any]],
    *,
    run_id: str,
    reference_metrics: Optional[Dict[str, Any]] = None,
    max_examples: int = 64,
) -> List[Dict[str, Any]]:
    """Build JSONL-ready offline teacher rows from a read-only channel snapshot."""

    snapshots = [
        feature_snapshot_from_channel_inventory(channel)
        for channel in channels
        if isinstance(channel, dict)
    ]
    context = build_inter_channel_context(snapshots)
    examples: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        recommendations = build_agentic_recommendations(snapshot, context=context)
        if not recommendations and snapshot.signal_activity <= 0.0:
            continue
        example = build_offline_teacher_example(
            run_id=run_id,
            snapshot=snapshot,
            recommendations=recommendations,
            reference_metrics=reference_metrics,
            operator_decision="unlabeled",
        ).to_dict()
        example["teacher_policy"] = {
            "source": FOUNDATION_VERSION,
            "offline_only": True,
            "live_write": False,
            "operator_decision_required_for_labels": True,
        }
        examples.append(example)
        if len(examples) >= max_examples:
            break
    return examples


def build_channel_foundation_summary(
    channel: Dict[str, Any],
    *,
    context: Optional[InterChannelContext] = None,
    config: Optional[AgenticFoundationConfig] = None,
) -> Dict[str, Any]:
    snapshot = feature_snapshot_from_channel_inventory(channel)
    recommendations = build_agentic_recommendations(snapshot, context=context, config=config)
    review = safety_review_recommendations(recommendations, config=config)
    return {
        "feature_snapshot": snapshot.to_dict(),
        "recommendations": [rec.to_dict() for rec in recommendations],
        "safety_review": {
            "summary": review["summary"],
            "blocked_reasons": sorted(
                {
                    reason
                    for row in review["blocked"]
                    for reason in row.get("blocked_reasons", [])
                }
            ),
        },
    }


def assert_no_live_transport_imports() -> bool:
    """Test helper: this module must stay independent from WING/OSC transports."""

    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "Wing" + "Client",
        "Enhanced" + "OSC" + "Client",
        "python" + "osc",
        "set" + "_fader",
        "set" + "_gain",
        "route" + "_output",
        "load" + "_snap",
        "save" + "_snap",
    )
    return not any(token in source for token in forbidden)
