"""Phase-aware EQ guard inspired by group phase-coordination plugins.

This module is transport-free: it does not import mixer clients and never sends
live delay, polarity, EQ, gain, or fader writes. It turns multichannel audio
evidence into proposal-only phase and EQ policy records.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload


MIN_FLOAT = 1e-12
DEFAULT_LOW_FREQ_RANGE_HZ = (35.0, 180.0)
DEFAULT_MAX_DELAY_MS = 10.0
PHASE_AWARE_EQ_GUARD_METHOD_ID = "phase_aware_eq_guard"
PHASE_AWARE_EQ_GUARD_VERSION = "automixer_phase_aware_eq_guard_v1"
PHASE_GUARD_READ_REQUIRED_SOURCE = (
    "fresh_multichannel_phase_audio_context_required_before_promotion"
)


def _proposal_denial_fields() -> Dict[str, Any]:
    return {
        "queue_action": "not_queueable_until_promoted",
        "can_import_to_queue": False,
        "can_apply_live": False,
        "queue_import_allowed": False,
        "live_apply_allowed": False,
        "read_only_or_dry_run_only": True,
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "live_write": False,
        "queue_mutation": False,
        "mutates_mixer": False,
    }


class GroupEnableMode(str, Enum):
    GROUP_FULL = "GROUP_FULL"
    GROUP_INTERNAL_ONLY = "GROUP_INTERNAL_ONLY"
    GROUP_PHASE_LOCK = "GROUP_PHASE_LOCK"
    GROUP_OFF = "GROUP_OFF"


class EQActionPolicy(str, Enum):
    PHASE_OK = "phase_ok"
    PHASE_RISK = "phase_risk"
    PHASE_CANDIDATE = "phase_candidate"
    PHASE_UNRELIABLE = "phase_unreliable"
    GROUP_DISABLED = "group_disabled"


@dataclass(frozen=True)
class PhaseAwareEQGuardConfig:
    sample_rate: int = 48000
    group_id: str = "default"
    group_enable_mode: GroupEnableMode = GroupEnableMode.GROUP_FULL
    global_enabled: bool = True
    low_freq_focus: bool = False
    low_freq_range_hz: Tuple[float, float] = DEFAULT_LOW_FREQ_RANGE_HZ
    lookahead_buffer_ms: float = 0.0
    report_latency: bool = True
    max_delay_ms: float = DEFAULT_MAX_DELAY_MS
    delay_candidate_threshold_ms: float = 0.35
    coherence_ok_threshold: float = 0.72
    coherence_risk_threshold: float = 0.42
    low_freq_loss_threshold_db: float = 2.0
    bleed_risk_threshold: float = 0.55
    anchor_weight_bias: float = 0.35
    dry_run: bool = True
    current_state_cache_freshness: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["group_enable_mode"] = self.group_enable_mode.value
        return data


@dataclass(frozen=True)
class PhaseChannelInput:
    channel_id: int
    audio: np.ndarray
    group_id: str = "default"
    weight: float = 1.0
    gain_db: float = 0.0
    label: str = ""
    bleed_ratio: float = 0.0
    muted: bool = False

    def mono_audio(self) -> np.ndarray:
        arr = np.asarray(self.audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return np.mean(arr, axis=1).astype(np.float32)
        return np.ravel(arr).astype(np.float32)


@dataclass(frozen=True)
class PhasePairMeasurement:
    anchor_channel: int
    candidate_channel: int
    delay_ms: float
    polarity_candidate: bool
    broadband_coherence: float
    low_freq_coherence: float
    low_freq_sum_loss_db: float
    phase_confidence: float
    candidate_weight: float
    candidate_bleed_ratio: float
    eq_action_policy: EQActionPolicy
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["eq_action_policy"] = self.eq_action_policy.value
        return data


@dataclass(frozen=True)
class PhaseAwareEQGuardResult:
    method_id: str
    group_id: str
    group_enable_mode: GroupEnableMode
    global_enabled: bool
    low_freq_focus: bool
    dry_run: bool
    sample_rate: int
    lookahead_buffer_ms: float
    report_latency: bool
    logic_latency_ms: float
    anchor_channel: Optional[int]
    group_gain_db: float
    eq_action_policy: EQActionPolicy
    pair_measurements: Tuple[PhasePairMeasurement, ...]
    is_group_graph: bool = True
    is_freq_graph: bool = True
    suggested_actions: Tuple[Dict[str, Any], ...] = ()
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["group_enable_mode"] = self.group_enable_mode.value
        data["eq_action_policy"] = self.eq_action_policy.value
        data["pair_measurements"] = [item.to_dict() for item in self.pair_measurements]
        data["suggested_actions"] = [dict(item) for item in self.suggested_actions]
        data["warnings"] = list(self.warnings)
        return data


def _rms(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))) + MIN_FLOAT)


def _amp_to_db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), MIN_FLOAT))


def _db_to_amp(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 20.0)


def _truncate_equal(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    length = min(int(a.size), int(b.size))
    if length <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.asarray(a[:length], dtype=np.float32), np.asarray(b[:length], dtype=np.float32)


def _gcc_phat_delay_ms(
    anchor: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_delay_ms: float,
) -> Tuple[float, float]:
    ref, sig = _truncate_equal(anchor, candidate)
    if ref.size < 16 or sig.size < 16 or _rms(ref) < 1e-8 or _rms(sig) < 1e-8:
        return 0.0, 0.0

    n = int(2 ** math.ceil(math.log2(max(ref.size, sig.size) * 2 - 1)))
    ref_fft = np.fft.rfft(ref, n=n)
    sig_fft = np.fft.rfft(sig, n=n)
    cross_power = np.conj(ref_fft) * sig_fft
    cross_power /= np.maximum(np.abs(cross_power), MIN_FLOAT)
    corr = np.fft.irfft(cross_power, n=n)
    corr = np.concatenate((corr[-(n // 2) :], corr[: n // 2]))
    center = corr.size // 2
    max_lag = min(int(abs(max_delay_ms) * sample_rate / 1000.0), center - 1)
    search = corr[center - max_lag : center + max_lag + 1]
    if search.size == 0:
        return 0.0, 0.0
    abs_search = np.abs(search)
    peak_index = int(np.argmax(abs_search))
    lag_samples = peak_index - max_lag
    peak_value = float(search[peak_index])
    return (lag_samples * 1000.0 / float(sample_rate)), peak_value


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _truncate_equal(a, b)
    if x.size < 2:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= MIN_FLOAT:
        return 0.0
    return float(np.dot(x, y) / denom)


def _band_limited(signal: np.ndarray, sample_rate: int, freq_range_hz: Tuple[float, float]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.size < 8:
        return x
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    spectrum = np.fft.rfft(x)
    lo_hz, hi_hz = freq_range_hz
    mask = (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
    filtered = np.zeros_like(spectrum)
    filtered[mask] = spectrum[mask]
    return np.fft.irfft(filtered, n=x.size).astype(np.float32)


def _coherence_score(a: np.ndarray, b: np.ndarray) -> float:
    return abs(_normalized_correlation(a, b))


def _low_freq_sum_loss_db(
    anchor: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    freq_range_hz: Tuple[float, float],
) -> float:
    ref, sig = _truncate_equal(anchor, candidate)
    ref_low = _band_limited(ref, sample_rate, freq_range_hz)
    sig_low = _band_limited(sig, sample_rate, freq_range_hz)
    separate_energy = _rms(ref_low) + _rms(sig_low)
    summed_energy = _rms(ref_low + sig_low)
    return max(0.0, _amp_to_db(separate_energy) - _amp_to_db(summed_energy))


def _select_anchor(channels: Sequence[PhaseChannelInput], config: PhaseAwareEQGuardConfig) -> PhaseChannelInput:
    def score(channel: PhaseChannelInput) -> float:
        level_score = _amp_to_db(_rms(channel.mono_audio())) + float(channel.gain_db)
        return level_score + (float(channel.weight) * 12.0 * float(config.anchor_weight_bias))

    return max(channels, key=score)


def _filter_channels(
    channels: Iterable[PhaseChannelInput],
    config: PhaseAwareEQGuardConfig,
) -> Tuple[PhaseChannelInput, ...]:
    usable = [channel for channel in channels if not channel.muted and channel.channel_id > 0]
    if config.group_enable_mode == GroupEnableMode.GROUP_INTERNAL_ONLY:
        usable = [channel for channel in usable if channel.group_id == config.group_id]
    return tuple(usable)


def _group_gain_db(channels: Sequence[PhaseChannelInput]) -> float:
    if not channels:
        return -120.0
    min_len = min(channel.mono_audio().size for channel in channels)
    if min_len <= 0:
        return -120.0
    summed = np.zeros(min_len, dtype=np.float32)
    for channel in channels:
        summed += channel.mono_audio()[:min_len] * _db_to_amp(channel.gain_db)
    return _amp_to_db(_rms(summed))


def _freshness_status(payload: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(payload, Mapping):
        return "not_provided"
    return str(payload.get("status") or "unknown").strip().lower() or "unknown"


def _freshness_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "not_provided", "provided": False}
    allowed_keys = (
        "status",
        "source",
        "field_count",
        "fresh_field_count",
        "stale_field_count",
        "unknown_field_count",
        "missing_field_count",
        "stale_fields",
        "missing_fields",
        "untimestamped_present_fields",
        "stale_after_sec",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload.get(key) for key in allowed_keys if key in payload}
    normalized["provided"] = True
    normalized.setdefault("status", _freshness_status(payload))
    return normalized


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def _base_metadata(
    config: PhaseAwareEQGuardConfig,
    *,
    candidate_count: int = 0,
    channel_count: int = 0,
    group_gain_source: str = "unavailable",
) -> Dict[str, Any]:
    freshness = _freshness_payload(config.current_state_cache_freshness)
    no_authority = _freshness_is_no_authority(freshness)
    return {
        "live_write_allowed": False,
        **_proposal_denial_fields(),
        "transport_free": True,
        "candidate_count": int(candidate_count),
        "channel_count": int(channel_count),
        "group_gain_source": group_gain_source,
        "current_state_cache_freshness_status": str(freshness.get("status") or "unknown"),
        "current_state_cache_freshness": freshness,
        "current_phase_audio_context_cache_no_live_authority": no_authority,
        "pi_surface_coverage": {
            "group": True,
            "group_enable_mode": True,
            "group_gain": True,
            "gain": True,
            "weight": True,
            "global_low_freq_focus": True,
            "lookahead_buffer_ms": True,
            "report_latency": True,
            "phase_lock": config.group_enable_mode == GroupEnableMode.GROUP_PHASE_LOCK,
        },
    }


def _pair_policy(
    *,
    delay_ms: float,
    polarity_candidate: bool,
    broadband_coherence: float,
    low_freq_coherence: float,
    low_freq_sum_loss_db: float,
    bleed_ratio: float,
    config: PhaseAwareEQGuardConfig,
) -> Tuple[EQActionPolicy, float, str]:
    coherence = low_freq_coherence if config.low_freq_focus else broadband_coherence
    phase_confidence = max(0.0, min(1.0, coherence * (1.0 - min(max(bleed_ratio, 0.0), 1.0) * 0.65)))

    if bleed_ratio >= config.bleed_risk_threshold and phase_confidence < config.coherence_risk_threshold:
        return (
            EQActionPolicy.PHASE_UNRELIABLE,
            phase_confidence,
            "high bleed and low phase confidence; hold EQ and request cleaner soundcheck evidence",
        )
    if low_freq_sum_loss_db >= config.low_freq_loss_threshold_db:
        return (
            EQActionPolicy.PHASE_CANDIDATE,
            phase_confidence,
            "low-frequency summing loss suggests phase cancellation before EQ",
        )
    if polarity_candidate and phase_confidence >= config.coherence_risk_threshold:
        return (
            EQActionPolicy.PHASE_CANDIDATE,
            phase_confidence,
            "negative correlation suggests a polarity candidate before EQ",
        )
    if abs(delay_ms) >= config.delay_candidate_threshold_ms and phase_confidence >= config.coherence_risk_threshold:
        return (
            EQActionPolicy.PHASE_CANDIDATE,
            phase_confidence,
            "measurable delay candidate should be reviewed before EQ",
        )
    if phase_confidence < config.coherence_ok_threshold:
        return (
            EQActionPolicy.PHASE_RISK,
            phase_confidence,
            "phase evidence is not clean enough for aggressive EQ",
        )
    return (
        EQActionPolicy.PHASE_OK,
        phase_confidence,
        "phase relationship is stable enough for normal EQ proposal flow",
    )


def _overall_policy(policies: Sequence[EQActionPolicy]) -> EQActionPolicy:
    priority = (
        EQActionPolicy.PHASE_CANDIDATE,
        EQActionPolicy.PHASE_UNRELIABLE,
        EQActionPolicy.PHASE_RISK,
        EQActionPolicy.PHASE_OK,
        EQActionPolicy.GROUP_DISABLED,
    )
    for item in priority:
        if item in policies:
            return item
    return EQActionPolicy.PHASE_UNRELIABLE


def _phase_eq_guard_safety_summary(
    measurement: PhasePairMeasurement,
    config: PhaseAwareEQGuardConfig,
    *,
    current_state_freshness_status: str,
    current_state_no_authority: bool,
) -> Dict[str, Any]:
    return {
        "summary_type": "phase_aware_eq_guard_review",
        "status": "candidate_review_required",
        "group_id": config.group_id,
        "phase_mode": config.group_enable_mode.value,
        "anchor_channel": measurement.anchor_channel,
        "candidate_channel": measurement.candidate_channel,
        "delay_ms": measurement.delay_ms,
        "polarity_candidate": measurement.polarity_candidate,
        "broadband_coherence": measurement.broadband_coherence,
        "low_freq_coherence": measurement.low_freq_coherence,
        "low_freq_sum_loss_db": measurement.low_freq_sum_loss_db,
        "phase_confidence": measurement.phase_confidence,
        "candidate_weight": measurement.candidate_weight,
        "candidate_bleed_ratio": measurement.candidate_bleed_ratio,
        "low_freq_focus": config.low_freq_focus,
        "eq_action_policy": measurement.eq_action_policy.value,
        "eq_hold_policy": "hold_aggressive_eq_until_phase_review",
        "current_phase_audio_context_cache_freshness": (
            current_state_freshness_status
        ),
        "current_phase_audio_context_cache_no_live_authority": (
            current_state_no_authority
        ),
        "group_phase_measurements": "present",
        "gcc_phat_confidence": measurement.phase_confidence,
        "sum_bus_check": "required_before_apply",
        "readback_required_before_promotion": True,
        "rollback_required_before_promotion": True,
        "operator_review_required": True,
        "live_write": False,
        "queue_mutation": False,
        **_proposal_denial_fields(),
        "can_authorize_live": False,
    }


def _suggested_action(
    measurement: PhasePairMeasurement,
    config: PhaseAwareEQGuardConfig,
    *,
    current_state_freshness_status: str,
    current_state_no_authority: bool,
) -> Optional[Dict[str, Any]]:
    if measurement.eq_action_policy != EQActionPolicy.PHASE_CANDIDATE:
        return None
    actions: List[str] = []
    if measurement.polarity_candidate:
        actions.append("polarity_candidate")
    if abs(measurement.delay_ms) >= config.delay_candidate_threshold_ms:
        actions.append("delay_candidate")
    if not actions and measurement.low_freq_sum_loss_db >= config.low_freq_loss_threshold_db:
        actions.append("low_freq_phase_review")

    safety_summary = _phase_eq_guard_safety_summary(
        measurement,
        config,
        current_state_freshness_status=current_state_freshness_status,
        current_state_no_authority=current_state_no_authority,
    )
    proposed_action = {
        "action_type": "phase_alignment_candidate",
        "dry_run": True,
        "anchor_channel": measurement.anchor_channel,
        "candidate_channel": measurement.candidate_channel,
        "phase_mode": config.group_enable_mode.value,
        "actions": actions,
        "delay_ms": measurement.delay_ms,
        "polarity_candidate": measurement.polarity_candidate,
        "low_freq_focus": config.low_freq_focus,
        "confidence": measurement.phase_confidence,
        "reason": measurement.reason,
        "phase_eq_guard_safety_summary": safety_summary,
    }
    evidence = {
        "current_phase_audio_context_cache_freshness": (
            current_state_freshness_status
        ),
        "current_phase_audio_context_cache_no_live_authority": (
            current_state_no_authority
        ),
        "group_phase_measurements": {
            "anchor_channel": measurement.anchor_channel,
            "candidate_channel": measurement.candidate_channel,
            "broadband_coherence": measurement.broadband_coherence,
            "low_freq_coherence": measurement.low_freq_coherence,
            "delay_ms": measurement.delay_ms,
            "polarity_candidate": measurement.polarity_candidate,
        },
        "gcc_phat_confidence": measurement.phase_confidence,
        "low_frequency_sum_loss": measurement.low_freq_sum_loss_db,
        "bleed_risk": measurement.candidate_bleed_ratio,
        "eq_hold_policy": "hold_aggressive_eq_until_phase_review",
        "phase_eq_guard_safety_summary": safety_summary,
        **_proposal_denial_fields(),
        "readback_required_before_promotion": True,
        "rollback_required_before_promotion": True,
    }
    blockers = [
        "unsupported_value_type:phase_eq_guard",
        "not_in_supervised_fader_gain_executor",
        "hardware_readback_and_rollback_not_proven",
    ]
    proposal = {
        "id": (
            f"{PHASE_AWARE_EQ_GUARD_METHOD_ID}:"
            f"{measurement.anchor_channel}:{measurement.candidate_channel}"
        ),
        "method_id": PHASE_AWARE_EQ_GUARD_METHOD_ID,
        "source": PHASE_AWARE_EQ_GUARD_METHOD_ID,
        "title": (
            f"Ch {measurement.candidate_channel}: phase guard before EQ proposal"
        ),
        "reason": measurement.reason,
        "target": measurement.candidate_channel,
        "kind": "phase_eq_guard",
        "severity": "medium",
        "confidence": round(max(0.0, min(1.0, measurement.phase_confidence)), 3),
        **_proposal_denial_fields(),
        "blockers": blockers,
        "proposed_action": proposed_action,
        "evidence": evidence,
    }
    action = {
        **proposed_action,
        "value_type": "phase_eq_guard",
        "target_kind": "group_phase_before_eq_policy",
        "live_write_allowed": False,
        **_proposal_denial_fields(),
        "evidence": evidence,
    }
    action["operator_correction"] = build_operator_correction_payload(
        proposal=proposal,
        method_id=PHASE_AWARE_EQ_GUARD_METHOD_ID,
        target=measurement.candidate_channel,
        proposed_action=proposed_action,
        blockers=blockers,
        value_type="phase_eq_guard",
        target_kind="group_phase_before_eq_policy",
        old_value_source=PHASE_GUARD_READ_REQUIRED_SOURCE,
    )
    return action


def analyze_phase_aware_eq_guard(
    channels: Sequence[PhaseChannelInput],
    config: PhaseAwareEQGuardConfig = PhaseAwareEQGuardConfig(),
) -> PhaseAwareEQGuardResult:
    """Build a proposal-only phase guard result for AutoEQ decisions."""

    logic_latency_ms = float(config.lookahead_buffer_ms) if config.report_latency else 0.0
    current_state_freshness = _freshness_payload(config.current_state_cache_freshness)
    current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
    current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
    if not config.global_enabled or config.group_enable_mode == GroupEnableMode.GROUP_OFF:
        return PhaseAwareEQGuardResult(
            method_id="automixer_phase_aware_eq_guard_v1",
            group_id=config.group_id,
            group_enable_mode=config.group_enable_mode,
            global_enabled=config.global_enabled,
            low_freq_focus=config.low_freq_focus,
            dry_run=True,
            sample_rate=config.sample_rate,
            lookahead_buffer_ms=config.lookahead_buffer_ms,
            report_latency=config.report_latency,
            logic_latency_ms=logic_latency_ms,
            anchor_channel=None,
            group_gain_db=-120.0,
            eq_action_policy=EQActionPolicy.GROUP_DISABLED,
            pair_measurements=(),
            suggested_actions=(),
            warnings=("group disabled; no phase or EQ action should be taken",),
            metadata=_base_metadata(config),
        )

    usable = _filter_channels(channels, config)
    if current_state_freshness_status != "fresh":
        metadata = _base_metadata(config, channel_count=len(usable))
        metadata["blockers"] = ("phase_aware_eq_context_cache_not_fresh",)
        return PhaseAwareEQGuardResult(
            method_id="automixer_phase_aware_eq_guard_v1",
            group_id=config.group_id,
            group_enable_mode=config.group_enable_mode,
            global_enabled=config.global_enabled,
            low_freq_focus=config.low_freq_focus,
            dry_run=True,
            sample_rate=config.sample_rate,
            lookahead_buffer_ms=config.lookahead_buffer_ms,
            report_latency=config.report_latency,
            logic_latency_ms=logic_latency_ms,
            anchor_channel=usable[0].channel_id if usable else None,
            group_gain_db=_group_gain_db(usable),
            eq_action_policy=EQActionPolicy.PHASE_UNRELIABLE,
            pair_measurements=(),
            suggested_actions=(),
            warnings=("phase context cache is not fresh; collect fresh phase evidence before EQ proposals",),
            metadata=metadata,
        )

    if not current_state_no_authority:
        metadata = _base_metadata(config, channel_count=len(usable))
        metadata["blockers"] = ("phase_aware_eq_context_cache_live_authority_present",)
        return PhaseAwareEQGuardResult(
            method_id="automixer_phase_aware_eq_guard_v1",
            group_id=config.group_id,
            group_enable_mode=config.group_enable_mode,
            global_enabled=config.global_enabled,
            low_freq_focus=config.low_freq_focus,
            dry_run=True,
            sample_rate=config.sample_rate,
            lookahead_buffer_ms=config.lookahead_buffer_ms,
            report_latency=config.report_latency,
            logic_latency_ms=logic_latency_ms,
            anchor_channel=usable[0].channel_id if usable else None,
            group_gain_db=_group_gain_db(usable),
            eq_action_policy=EQActionPolicy.PHASE_UNRELIABLE,
            pair_measurements=(),
            suggested_actions=(),
            warnings=(
                "phase context cache has live authority; collect read-only phase "
                "audio evidence before EQ proposals",
            ),
            metadata=metadata,
        )

    if len(usable) < 2:
        return PhaseAwareEQGuardResult(
            method_id="automixer_phase_aware_eq_guard_v1",
            group_id=config.group_id,
            group_enable_mode=config.group_enable_mode,
            global_enabled=config.global_enabled,
            low_freq_focus=config.low_freq_focus,
            dry_run=True,
            sample_rate=config.sample_rate,
            lookahead_buffer_ms=config.lookahead_buffer_ms,
            report_latency=config.report_latency,
            logic_latency_ms=logic_latency_ms,
            anchor_channel=usable[0].channel_id if usable else None,
            group_gain_db=_group_gain_db(usable),
            eq_action_policy=EQActionPolicy.PHASE_UNRELIABLE,
            pair_measurements=(),
            suggested_actions=(),
            warnings=("at least two usable channels are required for group phase analysis",),
            metadata=_base_metadata(config, channel_count=len(usable), group_gain_source="summed_weighted_audio"),
        )

    anchor = _select_anchor(usable, config)
    anchor_audio = anchor.mono_audio()
    measurements: List[PhasePairMeasurement] = []
    suggested_actions: List[Dict[str, Any]] = []

    for channel in usable:
        if channel.channel_id == anchor.channel_id:
            continue
        candidate_audio = channel.mono_audio()
        delay_ms, peak_value = _gcc_phat_delay_ms(anchor_audio, candidate_audio, config.sample_rate, config.max_delay_ms)
        broadband_coherence = _coherence_score(anchor_audio, candidate_audio)
        anchor_low = _band_limited(anchor_audio, config.sample_rate, config.low_freq_range_hz)
        candidate_low = _band_limited(candidate_audio, config.sample_rate, config.low_freq_range_hz)
        low_freq_coherence = _coherence_score(anchor_low, candidate_low)
        low_freq_loss = _low_freq_sum_loss_db(anchor_audio, candidate_audio, config.sample_rate, config.low_freq_range_hz)
        polarity_candidate = peak_value < 0.0 or _normalized_correlation(anchor_audio, candidate_audio) < -0.25
        policy, confidence, reason = _pair_policy(
            delay_ms=delay_ms,
            polarity_candidate=polarity_candidate,
            broadband_coherence=broadband_coherence,
            low_freq_coherence=low_freq_coherence,
            low_freq_sum_loss_db=low_freq_loss,
            bleed_ratio=channel.bleed_ratio,
            config=config,
        )

        measurement = PhasePairMeasurement(
            anchor_channel=anchor.channel_id,
            candidate_channel=channel.channel_id,
            delay_ms=delay_ms,
            polarity_candidate=polarity_candidate,
            broadband_coherence=broadband_coherence,
            low_freq_coherence=low_freq_coherence,
            low_freq_sum_loss_db=low_freq_loss,
            phase_confidence=confidence,
            candidate_weight=float(channel.weight),
            candidate_bleed_ratio=float(channel.bleed_ratio),
            eq_action_policy=policy,
            reason=reason,
        )
        measurements.append(measurement)
        action = _suggested_action(
            measurement,
            config,
            current_state_freshness_status=current_state_freshness_status,
            current_state_no_authority=current_state_no_authority,
        )
        if action:
            suggested_actions.append(action)

    overall = _overall_policy([measurement.eq_action_policy for measurement in measurements])
    warnings: List[str] = []
    if config.group_enable_mode == GroupEnableMode.GROUP_PHASE_LOCK:
        warnings.append("phase-lock mode: anchor channel is protected; only non-anchor channels get proposals")
    if overall in {EQActionPolicy.PHASE_CANDIDATE, EQActionPolicy.PHASE_RISK}:
        warnings.append("hold aggressive EQ boosts until phase candidates are reviewed")
    if overall == EQActionPolicy.PHASE_UNRELIABLE:
        warnings.append("phase evidence is unreliable; collect a longer or cleaner soundcheck window")

    return PhaseAwareEQGuardResult(
        method_id="automixer_phase_aware_eq_guard_v1",
        group_id=config.group_id,
        group_enable_mode=config.group_enable_mode,
        global_enabled=config.global_enabled,
        low_freq_focus=config.low_freq_focus,
        dry_run=True,
        sample_rate=config.sample_rate,
        lookahead_buffer_ms=config.lookahead_buffer_ms,
        report_latency=config.report_latency,
        logic_latency_ms=logic_latency_ms,
        anchor_channel=anchor.channel_id,
        group_gain_db=_group_gain_db(usable),
        eq_action_policy=overall,
        pair_measurements=tuple(measurements),
        suggested_actions=tuple(suggested_actions),
        warnings=tuple(warnings),
        metadata=_base_metadata(
            config,
            candidate_count=len(suggested_actions),
            channel_count=len(usable),
            group_gain_source="summed_weighted_audio",
        ),
    )


class PhaseAwareEQGuardPlugin:
    """Small plugin-style wrapper for stateful AutoMixer use."""

    def __init__(self, config: PhaseAwareEQGuardConfig = PhaseAwareEQGuardConfig()) -> None:
        self.config = config
        self.last_result: Optional[PhaseAwareEQGuardResult] = None

    def process(self, channels: Sequence[PhaseChannelInput]) -> PhaseAwareEQGuardResult:
        self.last_result = analyze_phase_aware_eq_guard(channels, self.config)
        return self.last_result

    def update_config(self, **overrides: Any) -> PhaseAwareEQGuardConfig:
        payload = self.config.to_dict()
        payload.update(overrides)
        if isinstance(payload.get("group_enable_mode"), str):
            payload["group_enable_mode"] = GroupEnableMode(payload["group_enable_mode"])
        self.config = PhaseAwareEQGuardConfig(**payload)
        return self.config
