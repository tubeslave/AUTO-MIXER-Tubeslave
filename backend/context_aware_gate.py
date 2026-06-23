"""Context-aware gate and bleed telemetry recommendations.

This module is deliberately transport-free. It never imports mixer clients and
never sends OSC/MIDI/live writes. It only builds dry-run recommendations and
decision-trace payloads that can be shown to the operator or stored for later
training/evaluation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from .automation_trace import AutomationDecisionTrace
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from automation_trace import AutomationDecisionTrace

try:
    from .source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust

try:
    from .processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - direct PYTHONPATH=backend compatibility
    from processing_proposal_adapters import build_operator_correction_payload


CONTEXT_AWARE_GATE_VERSION = "context_aware_gate_v1"
GATE_READ_REQUIRED_SOURCE = "cached_or_physical_gate_read_required_before_promotion"
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

INSTRUMENT_EXPECTED_BANDS = {
    "vocal": {"low_mid", "mid", "presence"},
    "speech": {"low_mid", "mid", "presence"},
    "kick": {"sub", "low"},
    "snare": {"low_mid", "mid", "presence"},
    "tom": {"low", "low_mid", "mid"},
    "floor_tom": {"low", "low_mid", "mid"},
    "overhead": {"presence", "air"},
    "cymbal": {"presence", "air"},
    "hihat": {"presence", "air"},
    "room": {"mid", "presence", "air"},
    "bass": {"sub", "low", "low_mid"},
    "guitar": {"low_mid", "mid", "presence"},
    "keys": {"low_mid", "mid", "presence", "air"},
}

DRUM_SOURCES = {"kick", "snare", "tom", "floor_tom", "rack_tom"}
TAIL_SENSITIVE_SOURCES = {"vocal", "speech", "guitar", "keys", "bass", "overhead", "room", "cymbal", "hihat"}
CYMBAL_CONTEXTS = {"overhead", "room", "cymbal", "hihat"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clip01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _dbfs(amplitude: float) -> float:
    if amplitude <= EPSILON:
        return -120.0
    return 20.0 * math.log10(max(float(amplitude), EPSILON))


def _amplitude_from_db(db_value: Any) -> float:
    try:
        numeric = float(db_value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return 10.0 ** (numeric / 20.0)


def _round(value: Any, digits: int = 3) -> float:
    return round(_safe_float(value), digits)


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
        "unknown_fields",
        "missing_fields",
        "max_age_seconds",
        "oldest_age_seconds",
        "read_only",
        "live_write",
        "queries_console",
    )
    normalized = {key: payload[key] for key in allowed_keys if key in payload}
    normalized.setdefault("status", _freshness_status(payload))
    normalized.setdefault("provided", True)
    return normalized


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def normalize_instrument(value: str) -> str:
    text = (value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lead_vocal": "vocal",
        "backing_vocal": "vocal",
        "vox": "vocal",
        "oh": "overhead",
        "oh_l": "overhead",
        "oh_r": "overhead",
        "hi_hat": "hihat",
        "hat": "hihat",
        "floor_tom": "floor_tom",
        "ftom": "floor_tom",
        "f_tom": "floor_tom",
        "electric_guitar": "guitar",
        "acoustic_guitar": "guitar",
        "bass_di": "bass",
        "bass_amp": "bass",
        "piano": "keys",
        "keyboard": "keys",
    }
    for needle, replacement in aliases.items():
        if needle in text:
            return replacement
    for known in INSTRUMENT_EXPECTED_BANDS:
        if known in text:
            return known
    return text or "unknown"


def _frame_rms_db(samples: np.ndarray, frame_size: int = 1024) -> list[float]:
    if samples.size == 0:
        return []
    if samples.size < frame_size:
        return [_dbfs(float(np.sqrt(np.mean(samples ** 2))))]
    frame_count = samples.size // frame_size
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return [_dbfs(float(value)) for value in rms]


def _frame_peak_db(samples: np.ndarray, frame_size: int = 1024) -> list[float]:
    if samples.size == 0:
        return []
    if samples.size < frame_size:
        return [_dbfs(float(np.max(np.abs(samples))))]
    frame_count = samples.size // frame_size
    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    peaks = np.max(np.abs(frames), axis=1)
    return [_dbfs(float(value)) for value in peaks]


def _bleed_correlation(samples: np.ndarray, peer_samples: Optional[Iterable[Sequence[float] | np.ndarray]]) -> float:
    if not peer_samples:
        return 0.0
    scores: list[float] = []
    for peer in peer_samples:
        peer_data = np.asarray(peer, dtype=float).reshape(-1)
        size = min(samples.size, peer_data.size)
        if size < 32:
            continue
        left = samples[:size]
        right = peer_data[:size]
        if np.std(left) <= EPSILON or np.std(right) <= EPSILON:
            continue
        scores.append(abs(float(np.corrcoef(left, right)[0, 1])))
    return _clip01(max(scores) if scores else 0.0)


def _band_energy_db(samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
    if samples.size < 32:
        return {band: -120.0 for band in FREQUENCY_BANDS}
    fft_size = min(max(2048, int(2 ** math.ceil(math.log2(min(samples.size, 8192))))), samples.size)
    frame = samples[:fft_size] * np.hanning(fft_size)
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    bands: Dict[str, float] = {}
    for band, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        energy = float(np.sqrt(np.mean(spectrum[mask] ** 2))) if np.any(mask) else 0.0
        bands[band] = _dbfs(energy / max(fft_size, 1))
    return bands


def spectral_outlier_score(expected_instrument: str, band_energy_db: Mapping[str, Any]) -> float:
    if not band_energy_db:
        return 0.0
    expected = INSTRUMENT_EXPECTED_BANDS.get(normalize_instrument(expected_instrument), {"low_mid", "mid", "presence"})
    expected_power = 0.0
    outlier_power = 0.0
    for band in FREQUENCY_BANDS:
        power = _amplitude_from_db(band_energy_db.get(band, -120.0))
        if band in expected:
            expected_power += power
        else:
            outlier_power += power
    return _clip01(outlier_power / max(expected_power + outlier_power, EPSILON))


def _transient_density(frame_rms: Sequence[float], frame_peaks: Sequence[float]) -> float:
    if not frame_rms or len(frame_rms) != len(frame_peaks):
        return 0.0
    crest_rows = [peak - rms for rms, peak in zip(frame_rms, frame_peaks)]
    return _clip01(sum(1 for crest in crest_rows if crest >= 8.0) / max(len(crest_rows), 1))


def _chatter_risk(frame_rms: Sequence[float], active_threshold_db: float) -> float:
    if len(frame_rms) < 4:
        return 0.0
    states = [level >= active_threshold_db for level in frame_rms]
    transitions = sum(1 for previous, current in zip(states, states[1:]) if previous != current)
    return _clip01(transitions / max(len(states) - 1, 1))


def _tail_risk(frame_rms: Sequence[float], active_threshold_db: float, instrument: str) -> float:
    source = normalize_instrument(instrument)
    if source not in TAIL_SENSITIVE_SOURCES or len(frame_rms) < 4:
        return 0.0
    peak_index = max(range(len(frame_rms)), key=lambda index: frame_rms[index])
    tail = frame_rms[peak_index + 1 :]
    if not tail:
        return 0.0
    lingering = sum(1 for level in tail if level >= active_threshold_db - 6.0) / len(tail)
    return _clip01(lingering)


@dataclass(frozen=True)
class BleedTelemetry:
    channel_id: int
    channel_label: str = ""
    expected_instrument: str = "unknown"
    noise_floor_db: float = -90.0
    activity_ratio: float = 0.0
    bleed_correlation_score: float = 0.0
    spectral_outlier_score: float = 0.0
    transient_density: float = 0.0
    tail_risk: float = 0.0
    chatter_risk: float = 0.0
    false_open_risk: float = 0.0
    false_close_risk: float = 0.0
    timestamp: str = field(default_factory=utc_now)
    source: str = CONTEXT_AWARE_GATE_VERSION
    current_state_cache_freshness: Optional[Dict[str, Any]] = None
    source_identity_trust_state: str = "not_provided"
    source_identity_trust_confidence: float = 0.0
    source_identity_replay_outcome_count: int = 0
    source_identity_calibration_freshness: Optional[Dict[str, Any]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_feature_snapshot(
        cls,
        snapshot: Any,
        *,
        channel_label: str = "",
        expected_instrument: Optional[str] = None,
    ) -> "BleedTelemetry":
        instrument = expected_instrument or getattr(snapshot, "role", "") or getattr(snapshot, "group", "") or "unknown"
        band_energy = getattr(snapshot, "band_energy_db", {}) or {}
        noise_floor = _safe_float(getattr(snapshot, "noise_floor_db", None), -90.0)
        activity = _safe_float(getattr(snapshot, "signal_activity", None), 0.0)
        bleed = _safe_float(getattr(snapshot, "bleed_index", None), 0.0)
        transient = _safe_float(getattr(snapshot, "transient_preservation_estimate", None), 0.0)
        false_open = _safe_float(getattr(snapshot, "false_open_risk", None), 0.0)
        false_close = _safe_float(getattr(snapshot, "false_close_risk", None), 0.0)
        chatter = _safe_float(getattr(snapshot, "chatter_risk", None), 0.0)
        return cls(
            channel_id=int(getattr(snapshot, "channel_id", 0) or 0),
            channel_label=channel_label or str(getattr(snapshot, "role", "") or f"CH {getattr(snapshot, 'channel_id', 0)}"),
            expected_instrument=normalize_instrument(str(instrument)),
            noise_floor_db=_round(noise_floor, 1),
            activity_ratio=_round(_clip01(activity), 3),
            bleed_correlation_score=_round(_clip01(bleed), 3),
            spectral_outlier_score=_round(spectral_outlier_score(str(instrument), band_energy), 3),
            transient_density=_round(_clip01(transient), 3),
            tail_risk=_round(_clip01(false_close), 3),
            chatter_risk=_round(_clip01(chatter), 3),
            false_open_risk=_round(_clip01(false_open), 3),
            false_close_risk=_round(_clip01(false_close), 3),
            current_state_cache_freshness=getattr(snapshot, "current_state_cache_freshness", None),
            source_identity_trust_state=str(getattr(snapshot, "source_identity_trust_state", "not_provided")),
            source_identity_trust_confidence=_safe_float(getattr(snapshot, "source_identity_trust_confidence", 0.0)),
            source_identity_replay_outcome_count=int(getattr(snapshot, "source_identity_replay_outcome_count", 0) or 0),
            source_identity_calibration_freshness=getattr(snapshot, "source_identity_calibration_freshness", None),
            evidence={
                "feature_source": getattr(snapshot, "source", "unknown"),
                "band_energy_db": dict(band_energy),
            },
        )


def extract_bleed_telemetry(
    samples: Sequence[float] | np.ndarray,
    *,
    channel_id: int,
    channel_label: str = "",
    expected_instrument: str = "unknown",
    peer_samples: Optional[Iterable[Sequence[float] | np.ndarray]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> BleedTelemetry:
    data = np.asarray(samples, dtype=float).reshape(-1)
    data = np.nan_to_num(data, copy=False)
    frame_rms = _frame_rms_db(data)
    frame_peaks = _frame_peak_db(data)
    noise_floor = float(np.percentile(frame_rms, 10)) if frame_rms else -120.0
    active_threshold = min(noise_floor + 6.0, -50.0)
    activity = (
        sum(1 for level in frame_rms if level >= active_threshold) / len(frame_rms)
        if frame_rms
        else 0.0
    )
    correlation = _bleed_correlation(data, peer_samples)
    bands = _band_energy_db(data, sample_rate)
    outlier = spectral_outlier_score(expected_instrument, bands)
    chatter = _chatter_risk(frame_rms, active_threshold)
    tail = _tail_risk(frame_rms, active_threshold, expected_instrument)
    false_open = _clip01(correlation * outlier * (1.0 - activity + 0.35))
    false_close = _clip01(tail + (0.35 * activity if normalize_instrument(expected_instrument) in TAIL_SENSITIVE_SOURCES else 0.0))
    return BleedTelemetry(
        channel_id=int(channel_id),
        channel_label=channel_label,
        expected_instrument=normalize_instrument(expected_instrument),
        noise_floor_db=_round(noise_floor, 1),
        activity_ratio=_round(activity, 3),
        bleed_correlation_score=_round(correlation, 3),
        spectral_outlier_score=_round(outlier, 3),
        transient_density=_round(_transient_density(frame_rms, frame_peaks), 3),
        tail_risk=_round(tail, 3),
        chatter_risk=_round(chatter, 3),
        false_open_risk=_round(false_open, 3),
        false_close_risk=_round(false_close, 3),
        evidence={"band_energy_db": bands, "sample_rate": int(sample_rate)},
    )


@dataclass(frozen=True)
class ContextGateRecommendation:
    channel_id: int
    agent_name: str = "ContextAwareGateAgent"
    action_type: str = "expander"
    suggested_params: Dict[str, Any] = field(default_factory=dict)
    current_value: Optional[float] = None
    confidence: float = 0.0
    risk_score: float = 1.0
    reason: str = ""
    safety_status: str = "proposal_only"
    dry_run: bool = True
    timestamp: str = field(default_factory=utc_now)
    idempotency_key: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["dry_run"] = True
        payload["idempotency_key"] = payload["idempotency_key"] or context_gate_idempotency_key(payload)
        return payload

    def gate_safety_summary(self) -> Dict[str, Any]:
        params = dict(self.suggested_params)
        chatter_guard = dict(self.evidence.get("chatter_guard") or {})
        quiet_words_tail_attack = dict(
            self.evidence.get("quiet_words_tail_attack_preservation") or {}
        )
        source_identity = dict(
            self.evidence.get("source_identity_downstream_trust") or {}
        )
        return {
            "summary_type": "context_aware_gate_review",
            "status": self.safety_status,
            "channel_id": self.channel_id,
            "action_type": self.action_type,
            "threshold_db": params.get("threshold_db"),
            "range_db": params.get("range_db"),
            "attack_ms": params.get("attack_ms"),
            "hold_ms": params.get("hold_ms"),
            "release_ms": params.get("release_ms"),
            "hysteresis_db": params.get("hysteresis_db"),
            "false_open_risk": self.evidence.get("false_open_risk"),
            "false_close_risk": self.evidence.get("false_close_risk"),
            "tail_risk": self.evidence.get("tail_risk"),
            "chatter_risk": self.evidence.get("chatter_risk"),
            "quiet_words_tail_attack_preservation": quiet_words_tail_attack,
            "quiet_words_tail_attack_status": quiet_words_tail_attack.get("status"),
            "chatter_guard_status": (
                "passed" if chatter_guard.get("hard_gate_allowed") else "blocked"
            ),
            "hard_gate_allowed": chatter_guard.get("hard_gate_allowed"),
            "source_identity_trust_status": source_identity.get("status"),
            "source_identity_usable_for_downstream": source_identity.get(
                "usable_for_downstream"
            ),
            "current_gate_bleed_context_cache_freshness": self.evidence.get(
                "current_gate_bleed_context_cache_freshness",
                self.evidence.get("current_state_cache_freshness_status"),
            ),
            "current_gate_bleed_context_cache_no_live_authority": self.evidence.get(
                "current_gate_bleed_context_cache_no_live_authority"
            ),
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
            "operator_review_required": True,
            "read_only_or_dry_run_only": True,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "can_authorize_live": False,
            "mutates_mixer": False,
        }

    def to_operator_proposal(self) -> Dict[str, Any]:
        """Return a Product Layer proposal without queue or live-apply authority."""

        payload = self.to_dict()
        params = dict(self.suggested_params)
        gate_safety_summary = self.gate_safety_summary()
        evidence = {
            **self.evidence,
            "bleed_telemetry": self.evidence.get("telemetry", {}),
            "current_gate_bleed_context_cache_freshness": self.evidence.get(
                "current_gate_bleed_context_cache_freshness",
                self.evidence.get("current_state_cache_freshness_status", "unknown"),
            ),
            "current_gate_bleed_context_cache_no_live_authority": self.evidence.get(
                "current_gate_bleed_context_cache_no_live_authority",
                False,
            ),
            "old_value_source": GATE_READ_REQUIRED_SOURCE,
            "rollback_value_source": GATE_READ_REQUIRED_SOURCE,
            "rollback_gate_readback": {
                "channel_id": self.channel_id,
                "mode": params.get("mode"),
                "threshold_db": params.get("threshold_db"),
                "range_db": params.get("range_db"),
                "attack_ms": params.get("attack_ms"),
                "hold_ms": params.get("hold_ms"),
                "release_ms": params.get("release_ms"),
                "hysteresis_db": params.get("hysteresis_db"),
                "source": GATE_READ_REQUIRED_SOURCE,
                "required_before_live_promotion": True,
                "live_write": False,
            },
            "gate_safety_summary": gate_safety_summary,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mutates_mixer": False,
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
        }
        evidence_blockers = self.evidence.get("blockers", ())
        if isinstance(evidence_blockers, str):
            evidence_blockers = (evidence_blockers,)
        blockers = [
            "unsupported_value_type:gate",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
            *tuple(evidence_blockers),
        ]
        proposed_action = {
            "action_type": self.action_type,
            "channel_id": self.channel_id,
            "params": params,
            "gate_safety_summary": gate_safety_summary,
        }
        proposal = {
            "id": payload["idempotency_key"],
            "method_id": "context_aware_gate_agent",
            "source": "context_aware_gate_agent",
            "title": f"Ch {self.channel_id}: gate/expander proposal",
            "reason": self.reason,
            "channel": self.channel_id,
            "target": self.channel_id,
            "kind": "gate",
            "severity": "medium" if self.safety_status == "review_required_high_risk" else "low",
            "confidence": self.confidence,
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
            "blockers": blockers,
            "proposed_action": proposed_action,
            "old_value": {
                "source": GATE_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "rollback_value": {
                "source": GATE_READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "evidence": evidence,
        }
        proposal["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id="context_aware_gate_agent",
            target=self.channel_id,
            proposed_action=proposed_action,
            blockers=blockers,
        )
        return proposal

    def to_decision_trace(self, *, mode: str = "dry_run") -> Dict[str, Any]:
        payload = self.to_dict()
        return AutomationDecisionTrace(
            idempotency_key=payload["idempotency_key"],
            proposal_id=payload["idempotency_key"],
            source=self.agent_name,
            mode=mode,
            target={"type": "channel", "id": self.channel_id},
            action_type=self.action_type,
            status=self.safety_status,
            current_value=self.current_value,
            proposed_value=self.suggested_params,
            delta=None,
            safety_limits={
                "dry_run": True,
                "live_write_allowed": False,
                "requires_supervised_gate": True,
                "blocked_live_action_types": ["gate", "expander"],
            },
            proof_status="dry_run_only",
            blocked_reason="gate_expander_live_write_quarantined",
            metadata={
                "confidence": self.confidence,
                "risk_score": self.risk_score,
                "reason": self.reason,
                "evidence": self.evidence,
            },
        ).to_dict()


def context_gate_idempotency_key(payload: Mapping[str, Any]) -> str:
    stable = "|".join(
        str(payload.get(key))
        for key in ("channel_id", "agent_name", "action_type", "suggested_params", "safety_status")
    )
    return f"context-gate-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class ContextAwareGateConfig:
    dry_run: bool = True
    live_writes_enabled: bool = False
    threshold_margin_db: float = 6.0
    min_bleed_score_for_recommendation: float = 0.35
    min_outlier_score_for_recommendation: float = 0.35
    high_risk_threshold: float = 0.68
    max_musical_range_db: float = 8.0
    max_drum_range_db: float = 24.0
    max_chatter_risk_for_hard_gate: float = 0.35
    max_false_close_risk_for_hard_gate: float = 0.45

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextAwareGateAgent:
    """Instrument-aware dry-run gate/expander recommendation agent."""

    def __init__(self, config: Optional[ContextAwareGateConfig] = None):
        self.config = config or ContextAwareGateConfig()

    def should_recommend(self, telemetry: BleedTelemetry) -> bool:
        if telemetry.bleed_correlation_score >= self.config.min_bleed_score_for_recommendation:
            return True
        if telemetry.spectral_outlier_score >= self.config.min_outlier_score_for_recommendation:
            return True
        if telemetry.false_open_risk >= 0.18 or telemetry.chatter_risk >= 0.25:
            return True
        return telemetry.noise_floor_db > -58.0

    def recommend(self, telemetry: BleedTelemetry) -> Optional[ContextGateRecommendation]:
        if not self.should_recommend(telemetry):
            return None

        instrument = normalize_instrument(telemetry.expected_instrument)
        current_state_freshness = _freshness_payload(telemetry.current_state_cache_freshness)
        current_state_freshness_status = str(current_state_freshness.get("status") or "unknown")
        current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
        source_identity_trust = evaluate_source_identity_downstream_trust(
            source_id=f"channel:{telemetry.channel_id}",
            source_type=instrument,
            trust_state=telemetry.source_identity_trust_state,
            trust_confidence=telemetry.source_identity_trust_confidence,
            replay_outcome_count=telemetry.source_identity_replay_outcome_count,
            current_state_cache_freshness=telemetry.source_identity_calibration_freshness,
        )
        blockers = []
        if current_state_freshness_status != "fresh":
            blockers.append("context_aware_gate_context_cache_not_fresh")
        elif not current_state_no_authority:
            blockers.append("context_aware_gate_context_cache_live_authority_present")
        blockers.extend(source_identity_trust.blockers)
        if blockers:
            blockers_tuple = tuple(sorted(set(blockers)))
            return ContextGateRecommendation(
                channel_id=telemetry.channel_id,
                action_type="gate_analysis_hold",
                suggested_params={},
                confidence=0.0,
                risk_score=1.0,
                reason=(
                    f"{telemetry.channel_label or 'Channel'} gate/expander proposal "
                    "is held until fresh context and trusted source identity are available."
                ),
                safety_status="analysis_only",
                dry_run=True,
                evidence={
                    "telemetry": telemetry.to_dict(),
                    "context_instrument": instrument,
                    "policy": self.config.to_dict(),
                    "current_state_cache_freshness_status": current_state_freshness_status,
                    "current_gate_bleed_context_cache_freshness": (
                        current_state_freshness_status
                    ),
                    "current_gate_bleed_context_cache_no_live_authority": (
                        current_state_no_authority
                    ),
                    "current_state_cache_freshness": current_state_freshness,
                    "source_identity_downstream_trust": source_identity_trust.to_dict(),
                    "blockers": blockers_tuple,
                    "old_value_source": GATE_READ_REQUIRED_SOURCE,
                    "rollback_value_source": GATE_READ_REQUIRED_SOURCE,
                    "readback_required_before_promotion": True,
                    "rollback_required_before_promotion": True,
                    "live_write": False,
                    "queue_mutation": False,
                    "model_mutation": False,
                    "memory_mutation": False,
                    "decision_log_ready": True,
                },
            )
        risk = self._risk_score(telemetry)
        confidence = self._confidence(telemetry, risk)
        chatter_guard = self._chatter_guard(telemetry)
        action_type = (
            "gate"
            if instrument in DRUM_SOURCES
            and telemetry.tail_risk < 0.55
            and chatter_guard["hard_gate_allowed"]
            else "expander"
        )
        safety_status = "proposal_only"
        if risk >= self.config.high_risk_threshold:
            safety_status = "review_required_high_risk"
        if not self.config.dry_run or self.config.live_writes_enabled:
            safety_status = "blocked_config_requires_dry_run"

        params = self._suggested_params(telemetry, instrument, action_type)
        return ContextGateRecommendation(
            channel_id=telemetry.channel_id,
            action_type=action_type,
            suggested_params=params,
            confidence=_round(confidence, 3),
            risk_score=_round(risk, 3),
            reason=self._reason(telemetry, instrument, action_type, risk),
            safety_status=safety_status,
            dry_run=True,
            evidence={
                "telemetry": telemetry.to_dict(),
                "context_instrument": instrument,
                "policy": self.config.to_dict(),
                "current_state_cache_freshness_status": current_state_freshness_status,
                "current_gate_bleed_context_cache_freshness": (
                    current_state_freshness_status
                ),
                "current_gate_bleed_context_cache_no_live_authority": (
                    current_state_no_authority
                ),
                "current_state_cache_freshness": current_state_freshness,
                "source_identity_downstream_trust": source_identity_trust.to_dict(),
                "false_open_risk": telemetry.false_open_risk,
                "false_close_risk": telemetry.false_close_risk,
                "chatter_risk": telemetry.chatter_risk,
                "tail_risk": telemetry.tail_risk,
                "quiet_words_tail_attack_preservation": {
                    "status": "operator_review_required_before_live_promotion",
                    "tail_risk": telemetry.tail_risk,
                    "false_close_risk": telemetry.false_close_risk,
                    "attack_ms": params.get("attack_ms"),
                    "hold_ms": params.get("hold_ms"),
                    "release_ms": params.get("release_ms"),
                    "hard_gate_allowed": chatter_guard.get("hard_gate_allowed"),
                },
                "chatter_guard": chatter_guard,
                "hysteresis": params.get("hysteresis_db"),
                "old_value_source": GATE_READ_REQUIRED_SOURCE,
                "rollback_value_source": GATE_READ_REQUIRED_SOURCE,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
                "live_write": False,
                "queue_mutation": False,
                "model_mutation": False,
                "memory_mutation": False,
                "decision_log_ready": True,
            },
        )

    def _suggested_params(self, telemetry: BleedTelemetry, instrument: str, action_type: str) -> Dict[str, Any]:
        threshold = round(telemetry.noise_floor_db + self.config.threshold_margin_db, 1)
        if instrument in CYMBAL_CONTEXTS:
            return {
                "mode": "expander",
                "threshold_db": threshold,
                "range_db": -3.0,
                "attack_ms": 3.0,
                "hold_ms": 220.0,
                "release_ms": 420.0,
                "hysteresis_db": 4.0,
            }
        if instrument in DRUM_SOURCES and action_type == "gate":
            return {
                "mode": "gate",
                "threshold_db": threshold + 2.0,
                "range_db": -min(self.config.max_drum_range_db, 18.0 + 6.0 * telemetry.bleed_correlation_score),
                "attack_ms": 0.5,
                "hold_ms": 140.0,
                "release_ms": 180.0,
                "hysteresis_db": 3.0,
            }
        return {
            "mode": "expander",
            "threshold_db": threshold,
            "range_db": -min(self.config.max_musical_range_db, 4.0 + 6.0 * telemetry.bleed_correlation_score),
            "attack_ms": 2.0,
            "hold_ms": 180.0,
            "release_ms": 280.0 if telemetry.tail_risk < 0.5 else 420.0,
            "hysteresis_db": 4.0,
        }

    def _chatter_guard(self, telemetry: BleedTelemetry) -> Dict[str, Any]:
        max_chatter = _clip01(self.config.max_chatter_risk_for_hard_gate)
        chatter_risk = _clip01(telemetry.chatter_risk)
        max_false_close = _clip01(self.config.max_false_close_risk_for_hard_gate)
        false_close_risk = _clip01(telemetry.false_close_risk)
        return {
            "chatter_risk": _round(chatter_risk, 3),
            "max_chatter_risk_for_hard_gate": _round(max_chatter, 3),
            "false_close_risk": _round(false_close_risk, 3),
            "max_false_close_risk_for_hard_gate": _round(max_false_close, 3),
            "hard_gate_allowed": (
                chatter_risk <= max_chatter
                and false_close_risk <= max_false_close
            ),
        }

    def _risk_score(self, telemetry: BleedTelemetry) -> float:
        return _clip01(
            0.28 * telemetry.false_close_risk
            + 0.22 * telemetry.tail_risk
            + 0.20 * telemetry.chatter_risk
            + 0.18 * telemetry.false_open_risk
            + 0.12 * telemetry.spectral_outlier_score
        )

    def _confidence(self, telemetry: BleedTelemetry, risk: float) -> float:
        return _clip01(
            0.42
            + 0.26 * telemetry.bleed_correlation_score
            + 0.18 * telemetry.spectral_outlier_score
            + 0.08 * telemetry.activity_ratio
            - 0.22 * risk
        )

    def _reason(self, telemetry: BleedTelemetry, instrument: str, action_type: str, risk: float) -> str:
        if instrument in CYMBAL_CONTEXTS:
            return (
                f"{telemetry.channel_label or 'Channel'} is expected to carry cymbal/room content, "
                "so the agent avoids hard gating and only recommends a gentle expander review."
            )
        if action_type == "gate":
            return (
                f"{telemetry.channel_label or 'Channel'} looks like a transient drum source with contextual bleed; "
                "recommend a dry-run gate proposal for operator review."
            )
        risk_note = " High tail/false-close risk keeps this in review." if risk >= self.config.high_risk_threshold else ""
        return (
            f"{telemetry.channel_label or 'Channel'} has out-of-context bleed/noise for expected {instrument}; "
            f"recommend a gentle dry-run expander proposal, not a threshold-only gate.{risk_note}"
        )


def recommendation_to_decision_trace(recommendation: ContextGateRecommendation, *, mode: str = "dry_run") -> Dict[str, Any]:
    return recommendation.to_decision_trace(mode=mode)


def build_context_gate_proposals(
    telemetries: Iterable[BleedTelemetry],
    *,
    agent: Optional[ContextAwareGateAgent] = None,
) -> tuple[Dict[str, Any], ...]:
    gate_agent = agent or ContextAwareGateAgent()
    proposals = []
    for telemetry in telemetries:
        recommendation = gate_agent.recommend(telemetry)
        if recommendation is None:
            continue
        if not recommendation.suggested_params:
            continue
        if recommendation.safety_status not in {"proposal_only", "review_required_high_risk"}:
            continue
        proposals.append(recommendation.to_operator_proposal())
    return tuple(proposals)


def assert_no_live_transport_imports() -> bool:
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
