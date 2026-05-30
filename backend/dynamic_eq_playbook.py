"""Shadow-only bridge for dynamic EQ model predictions.

The module accepts external neural dynamic-EQ predictions, such as six-band
PEQ+DRC outputs inspired by Diff-DEQ research, validates them against Automixer
safety limits and turns them into dry-run playbook entries. It never imports
mixer clients and never sends live EQ/dynamics writes.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised through package and PYTHONPATH modes
    from autoeq_audit import AutoEQAuditConfig, ParametricEQBand, audit_peq_bands
except ImportError:  # pragma: no cover
    from backend.autoeq_audit import AutoEQAuditConfig, ParametricEQBand, audit_peq_bands


SCHEMA_VERSION = "0.1"
DEFAULT_MODEL_FAMILY = "external_dynamic_eq"
DEFAULT_EXPECTED_BAND_COUNT = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pick(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


@dataclass(frozen=True)
class DynamicEQSafetyConfig:
    """Safe bounds for shadow dynamic-EQ predictions."""

    expected_band_count: int = DEFAULT_EXPECTED_BAND_COUNT
    max_abs_eq_gain_db: float = 6.0
    max_q: float = 10.0
    max_dynamic_gain_range_db: float = 8.0
    max_makeup_gain_db: float = 3.0
    min_threshold_db: float = -60.0
    max_threshold_db: float = 0.0
    min_ratio: float = 1.0
    max_ratio: float = 10.0
    min_attack_ms: float = 0.1
    max_attack_ms: float = 250.0
    min_release_ms: float = 5.0
    max_release_ms: float = 4000.0
    confidence_threshold: float = 0.65
    live_write_allowed: bool = False
    dry_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_autoeq_config(self) -> AutoEQAuditConfig:
        return AutoEQAuditConfig(max_abs_gain_db=self.max_abs_eq_gain_db, max_q=self.max_q)


@dataclass(frozen=True)
class DynamicEQIssue:
    code: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicEQBandPrediction:
    band_index: int
    center_hz: float
    q: float
    peq_gain_db: float = 0.0
    threshold_db: Optional[float] = None
    ratio: Optional[float] = None
    attack_ms: Optional[float] = None
    release_ms: Optional[float] = None
    dynamic_gain_range_db: Optional[float] = None
    makeup_gain_db: float = 0.0
    label: str = ""

    def to_peq_band(self) -> ParametricEQBand:
        return ParametricEQBand(
            frequency_hz=self.center_hz,
            gain_db=self.peq_gain_db,
            q=self.q,
            band_type="peaking",
            label=self.label or f"dynamic_eq_band_{self.band_index}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicEQPrediction:
    channel_id: int
    source_model: str = DEFAULT_MODEL_FAMILY
    model_version: str = "unknown"
    instrument_role: str = "unknown"
    sample_rate: int = 48000
    window_start_ms: Optional[float] = None
    window_duration_ms: Optional[float] = None
    confidence: float = 0.0
    bands: Tuple[DynamicEQBandPrediction, ...] = ()
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bands"] = [band.to_dict() for band in self.bands]
        return payload


@dataclass(frozen=True)
class DynamicEQValidationResult:
    passed: bool
    issues: Tuple[DynamicEQIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DynamicEQPlaybookEntry:
    recommendation_id: str
    channel_id: int
    agent_name: str
    action_type: str
    safety_status: str
    dry_run: bool
    confidence: float
    risk_score: float
    reason: str
    suggested_params: Dict[str, Any]
    validation: Dict[str, Any]
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_dynamic_eq_prediction(payload: Mapping[str, Any]) -> DynamicEQPrediction:
    """Parse a JSON-like external dynamic-EQ payload into a typed prediction."""

    bands: List[DynamicEQBandPrediction] = []
    for index, raw_band in enumerate(payload.get("bands", []) or [], start=1):
        if not isinstance(raw_band, Mapping):
            continue
        center_hz = _safe_float(_pick(raw_band, "center_hz", "frequency_hz", "freq_hz", "f0_hz"))
        q = _safe_float(_pick(raw_band, "q", "Q", "quality"))
        if center_hz is None or q is None:
            continue
        band_index = _safe_int(_pick(raw_band, "band_index", "index", default=index), default=index)
        bands.append(
            DynamicEQBandPrediction(
                band_index=band_index,
                center_hz=center_hz,
                q=q,
                peq_gain_db=float(_safe_float(_pick(raw_band, "peq_gain_db", "gain_db", "eq_gain_db", default=0.0)) or 0.0),
                threshold_db=_safe_float(_pick(raw_band, "threshold_db", "drc_threshold_db")),
                ratio=_safe_float(_pick(raw_band, "ratio", "drc_ratio")),
                attack_ms=_safe_float(_pick(raw_band, "attack_ms", "drc_attack_ms")),
                release_ms=_safe_float(_pick(raw_band, "release_ms", "drc_release_ms")),
                dynamic_gain_range_db=_safe_float(_pick(raw_band, "dynamic_gain_range_db", "range_db", "drc_range_db")),
                makeup_gain_db=float(_safe_float(_pick(raw_band, "makeup_gain_db", "drc_makeup_gain_db", default=0.0)) or 0.0),
                label=str(_pick(raw_band, "label", "name", default=f"band_{band_index}")),
            )
        )

    return DynamicEQPrediction(
        channel_id=_safe_int(_pick(payload, "channel_id", "channel", default=0)),
        source_model=str(_pick(payload, "source_model", "model", default=DEFAULT_MODEL_FAMILY)),
        model_version=str(_pick(payload, "model_version", "version", default="unknown")),
        instrument_role=str(_pick(payload, "instrument_role", "role", default="unknown")),
        sample_rate=_safe_int(_pick(payload, "sample_rate", "sample_rate_hz", default=48000), default=48000),
        window_start_ms=_safe_float(_pick(payload, "window_start_ms", "start_ms")),
        window_duration_ms=_safe_float(_pick(payload, "window_duration_ms", "duration_ms")),
        confidence=float(_safe_float(_pick(payload, "confidence", default=0.0)) or 0.0),
        bands=tuple(bands),
        source_metadata=dict(payload.get("source_metadata", {}) or {}),
    )


def validate_dynamic_eq_prediction(
    prediction: DynamicEQPrediction,
    config: DynamicEQSafetyConfig = DynamicEQSafetyConfig(),
) -> DynamicEQValidationResult:
    """Validate an external dynamic-EQ prediction for shadow/dry-run use."""

    issues: List[DynamicEQIssue] = []

    if prediction.channel_id <= 0:
        issues.append(
            DynamicEQIssue(
                code="invalid_channel_id",
                severity="error",
                message="Dynamic EQ prediction must reference a positive channel id.",
                details={"channel_id": prediction.channel_id},
            )
        )
    if prediction.sample_rate <= 0:
        issues.append(
            DynamicEQIssue(
                code="invalid_sample_rate",
                severity="error",
                message="Dynamic EQ prediction must record a positive sample rate.",
                details={"sample_rate": prediction.sample_rate},
            )
        )
    if len(prediction.bands) != config.expected_band_count:
        issues.append(
            DynamicEQIssue(
                code="unexpected_band_count",
                severity="warning",
                message="Dynamic EQ prediction band count differs from the expected model contract.",
                details={"band_count": len(prediction.bands), "expected": config.expected_band_count},
            )
        )
    if prediction.confidence < config.confidence_threshold:
        issues.append(
            DynamicEQIssue(
                code="low_model_confidence",
                severity="warning",
                message="Dynamic EQ prediction confidence is below the review threshold.",
                details={"confidence": prediction.confidence, "threshold": config.confidence_threshold},
            )
        )

    peq_result = audit_peq_bands([band.to_peq_band() for band in prediction.bands], config=config.as_autoeq_config())
    for issue in peq_result.issues:
        issues.append(DynamicEQIssue(issue.code, issue.severity, issue.message, issue.details))

    for band in prediction.bands:
        label = band.label or f"band_{band.band_index}"
        if band.dynamic_gain_range_db is not None and abs(band.dynamic_gain_range_db) > config.max_dynamic_gain_range_db:
            issues.append(
                DynamicEQIssue(
                    code="dynamic_gain_range_limit_exceeded",
                    severity="error",
                    message="Dynamic EQ band gain range exceeds safety limit.",
                    details={"label": label, "range_db": band.dynamic_gain_range_db, "limit_db": config.max_dynamic_gain_range_db},
                )
            )
        if abs(band.makeup_gain_db) > config.max_makeup_gain_db:
            issues.append(
                DynamicEQIssue(
                    code="makeup_gain_limit_exceeded",
                    severity="error",
                    message="Dynamic EQ/DRC makeup gain exceeds safety limit.",
                    details={"label": label, "makeup_gain_db": band.makeup_gain_db, "limit_db": config.max_makeup_gain_db},
                )
            )
        if band.threshold_db is not None and not (config.min_threshold_db <= band.threshold_db <= config.max_threshold_db):
            issues.append(
                DynamicEQIssue(
                    code="threshold_out_of_range",
                    severity="error",
                    message="DRC threshold is outside the allowed range.",
                    details={
                        "label": label,
                        "threshold_db": band.threshold_db,
                        "allowed": [config.min_threshold_db, config.max_threshold_db],
                    },
                )
            )
        if band.ratio is not None and not (config.min_ratio <= band.ratio <= config.max_ratio):
            issues.append(
                DynamicEQIssue(
                    code="ratio_out_of_range",
                    severity="error",
                    message="DRC ratio is outside the allowed range.",
                    details={"label": label, "ratio": band.ratio, "allowed": [config.min_ratio, config.max_ratio]},
                )
            )
        if band.attack_ms is not None and not (config.min_attack_ms <= band.attack_ms <= config.max_attack_ms):
            issues.append(
                DynamicEQIssue(
                    code="attack_out_of_range",
                    severity="error",
                    message="DRC attack is outside the allowed range.",
                    details={"label": label, "attack_ms": band.attack_ms, "allowed": [config.min_attack_ms, config.max_attack_ms]},
                )
            )
        if band.release_ms is not None and not (config.min_release_ms <= band.release_ms <= config.max_release_ms):
            issues.append(
                DynamicEQIssue(
                    code="release_out_of_range",
                    severity="error",
                    message="DRC release is outside the allowed range.",
                    details={"label": label, "release_ms": band.release_ms, "allowed": [config.min_release_ms, config.max_release_ms]},
                )
            )

    return DynamicEQValidationResult(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "source_model": prediction.source_model,
            "model_version": prediction.model_version,
            "band_count": len(prediction.bands),
            "dry_run": config.dry_run,
            "live_write_allowed": config.live_write_allowed,
        },
    )


def dynamic_eq_idempotency_key(prediction: DynamicEQPrediction, validation: DynamicEQValidationResult) -> str:
    stable = "|".join(
        [
            prediction.source_model,
            prediction.model_version,
            str(prediction.channel_id),
            str(round(prediction.confidence, 4)),
            str(len(prediction.bands)),
            str(validation.passed),
        ]
    )
    return f"dynamic-eq-{hashlib.sha256(stable.encode('utf-8')).hexdigest()[:16]}"


def build_dynamic_eq_playbook_entry(
    prediction: DynamicEQPrediction,
    config: DynamicEQSafetyConfig = DynamicEQSafetyConfig(),
) -> DynamicEQPlaybookEntry:
    """Create a dry-run playbook entry from a validated dynamic-EQ prediction."""

    validation = validate_dynamic_eq_prediction(prediction, config=config)
    error_count = sum(1 for issue in validation.issues if issue.severity == "error")
    warning_count = sum(1 for issue in validation.issues if issue.severity == "warning")
    if error_count:
        safety_status = "blocked_shadow_only"
        reason = "Dynamic EQ prediction failed safety validation; no live action is allowed."
    elif warning_count:
        safety_status = "review_shadow_only"
        reason = "Dynamic EQ prediction is structurally valid but needs operator/research review."
    else:
        safety_status = "shadow_only"
        reason = "Dynamic EQ prediction passed offline safety validation; keep as research playbook."

    risk_score = min(1.0, max(0.0, (error_count * 0.5) + (warning_count * 0.15) + (1.0 - prediction.confidence) * 0.25))
    return DynamicEQPlaybookEntry(
        recommendation_id=dynamic_eq_idempotency_key(prediction, validation),
        channel_id=prediction.channel_id,
        agent_name="DynamicEQPlaybookBridge",
        action_type="dynamic_eq_playbook",
        safety_status=safety_status,
        dry_run=True,
        confidence=max(0.0, min(1.0, prediction.confidence)),
        risk_score=round(risk_score, 3),
        reason=reason,
        suggested_params={
            "source_model": prediction.source_model,
            "model_version": prediction.model_version,
            "instrument_role": prediction.instrument_role,
            "sample_rate": prediction.sample_rate,
            "window_start_ms": prediction.window_start_ms,
            "window_duration_ms": prediction.window_duration_ms,
            "bands": [band.to_dict() for band in prediction.bands],
        },
        validation=validation.to_dict(),
    )


def build_dynamic_eq_playbook(
    payloads: Iterable[Mapping[str, Any]],
    config: DynamicEQSafetyConfig = DynamicEQSafetyConfig(),
) -> Dict[str, Any]:
    """Parse, validate and summarize a collection of external predictions."""

    entries = [build_dynamic_eq_playbook_entry(parse_dynamic_eq_prediction(payload), config=config) for payload in payloads]
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "dynamic_eq_playbook",
        "dry_run": True,
        "live_write_allowed": False,
        "entry_count": len(entries),
        "blocked_count": sum(1 for entry in entries if entry.safety_status == "blocked_shadow_only"),
        "review_count": sum(1 for entry in entries if entry.safety_status == "review_shadow_only"),
        "entries": [entry.to_dict() for entry in entries],
    }
