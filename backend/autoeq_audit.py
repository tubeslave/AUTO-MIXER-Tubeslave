"""Audit-only helpers for AutoEQ filter packages.

This module validates EQ targets, filters, dataset splits, export consistency
and gain staging before any EQ recommendation can be promoted. It is deliberately
transport-free: no WING, OSC, server or handler imports belong here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "0.1"
MIN_LINEAR_MAGNITUDE = 1e-12
VALID_PHASE_TYPES = {"minimum", "linear"}
PHASE_ALIASES = {
    "min": "minimum",
    "minimum": "minimum",
    "minimum_phase": "minimum",
    "minimum-phase": "minimum",
    "linear": "linear",
    "linear_phase": "linear",
    "linear-phase": "linear",
}


@dataclass(frozen=True)
class AutoEQAuditConfig:
    """Safety limits for validating AutoEQ filters and reports."""

    max_abs_gain_db: float = 12.0
    max_q: float = 10.0
    max_pole_radius: float = 0.999
    max_narrow_peaks: int = 0
    narrow_peak_threshold_db: float = 12.0
    narrow_peak_prominence_db: float = 6.0
    true_peak_margin_db: float = 3.0
    export_tolerance_db: float = 0.2
    max_total_positive_gain_db: float = 9.0


@dataclass(frozen=True)
class AutoEQAuditIssue:
    """Single validation issue."""

    code: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class AutoEQAuditResult:
    """Validation result with machine-readable issues."""

    passed: bool
    issues: Tuple[AutoEQAuditIssue, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PhaseContract:
    """Explicit phase assumptions for an AutoEQ target/filter pair."""

    target_phase: str
    filter_phase: str
    target_source: str = "unknown"
    filter_design: str = "unknown"
    minimum_phase_reconstruction: bool = False

    def normalized(self) -> "PhaseContract":
        return replace(
            self,
            target_phase=normalize_phase_type(self.target_phase),
            filter_phase=normalize_phase_type(self.filter_phase),
        )


@dataclass(frozen=True)
class ParametricEQBand:
    """Parametric EQ band description used by audit checks."""

    frequency_hz: float
    gain_db: float
    q: float
    band_type: str = "peaking"
    label: str = ""

    def limited(self, config: AutoEQAuditConfig = AutoEQAuditConfig()) -> "ParametricEQBand":
        return replace(
            self,
            gain_db=float(np.clip(self.gain_db, -config.max_abs_gain_db, config.max_abs_gain_db)),
            q=float(min(max(self.q, 0.001), config.max_q)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BiquadCoefficients:
    """Second-order IIR section with explicit EQ metadata."""

    b0: float
    b1: float
    b2: float
    a0: float
    a1: float
    a2: float
    gain_db: float = 0.0
    q: Optional[float] = None
    label: str = ""

    def normalized(self) -> "BiquadCoefficients":
        if abs(self.a0) <= MIN_LINEAR_MAGNITUDE:
            raise ValueError("a0 must be non-zero for a stable biquad")
        if abs(self.a0 - 1.0) <= 1e-12:
            return self
        return BiquadCoefficients(
            b0=self.b0 / self.a0,
            b1=self.b1 / self.a0,
            b2=self.b2 / self.a0,
            a0=1.0,
            a1=self.a1 / self.a0,
            a2=self.a2 / self.a0,
            gain_db=self.gain_db,
            q=self.q,
            label=self.label,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeadroomAdjustment:
    """Result of true-peak headroom normalization."""

    true_peak_before_dbfs: float
    true_peak_after_dbfs: float
    applied_gain_db: float
    margin_db: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def db_to_lin(gain_db: float | np.ndarray) -> float | np.ndarray:
    """Convert dB gain to linear magnitude."""

    return np.power(10.0, np.asarray(gain_db, dtype=float) / 20.0)


def lin_to_db(mag_lin: float | np.ndarray) -> float | np.ndarray:
    """Convert linear magnitude to dB gain with a safe floor."""

    magnitude = np.maximum(np.asarray(mag_lin, dtype=float), MIN_LINEAR_MAGNITUDE)
    return 20.0 * np.log10(magnitude)


def normalize_phase_type(phase_type: str) -> str:
    """Normalize supported phase labels to ``minimum`` or ``linear``."""

    normalized = PHASE_ALIASES.get(str(phase_type).strip().lower())
    if normalized not in VALID_PHASE_TYPES:
        raise ValueError(f"unsupported AutoEQ phase type: {phase_type!r}")
    return normalized


def validate_phase_contract(contract: PhaseContract) -> AutoEQAuditResult:
    """Validate that target and filter phase assumptions are explicit and compatible."""

    issues: List[AutoEQAuditIssue] = []
    try:
        normalized = contract.normalized()
    except ValueError as exc:
        return AutoEQAuditResult(
            passed=False,
            issues=(
                AutoEQAuditIssue(
                    code="invalid_phase_type",
                    severity="error",
                    message=str(exc),
                    details={"target_phase": contract.target_phase, "filter_phase": contract.filter_phase},
                ),
            ),
        )

    if normalized.target_phase != normalized.filter_phase:
        issues.append(
            AutoEQAuditIssue(
                code="phase_contract_mismatch",
                severity="error",
                message="AutoEQ target phase and filter phase must match before promotion.",
                details={
                    "target_phase": normalized.target_phase,
                    "filter_phase": normalized.filter_phase,
                    "target_source": normalized.target_source,
                    "filter_design": normalized.filter_design,
                },
            )
        )

    if normalized.target_phase == "minimum" and not normalized.minimum_phase_reconstruction:
        issues.append(
            AutoEQAuditIssue(
                code="minimum_phase_reconstruction_unverified",
                severity="warning",
                message="Minimum-phase target should record how magnitude-only data was reconstructed.",
                details={"target_source": normalized.target_source},
            )
        )

    return AutoEQAuditResult(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        metadata={"phase_contract": asdict(normalized), "schema_version": SCHEMA_VERSION},
    )


def biquad_poles(biquad: BiquadCoefficients) -> np.ndarray:
    """Return normalized biquad poles."""

    bq = biquad.normalized()
    return np.roots([1.0, bq.a1, bq.a2])


def check_biquad_stability(
    biquad: BiquadCoefficients,
    max_pole_radius: float = AutoEQAuditConfig.max_pole_radius,
) -> bool:
    """Return True when all poles are safely inside the unit circle."""

    poles = biquad_poles(biquad)
    return bool(np.all(np.abs(poles) < max_pole_radius))


def enforce_peq_limits(
    peq_bands: Sequence[ParametricEQBand],
    config: AutoEQAuditConfig = AutoEQAuditConfig(),
) -> Tuple[ParametricEQBand, ...]:
    """Return PEQ bands clipped to the audit gain/Q budget."""

    return tuple(band.limited(config) for band in peq_bands)


def audit_peq_bands(
    peq_bands: Sequence[ParametricEQBand],
    config: AutoEQAuditConfig = AutoEQAuditConfig(),
) -> AutoEQAuditResult:
    """Validate gain/Q/frequency limits for parametric EQ bands."""

    issues: List[AutoEQAuditIssue] = []
    total_positive_gain = 0.0
    for index, band in enumerate(peq_bands):
        label = band.label or f"band_{index + 1}"
        if band.frequency_hz <= 0:
            issues.append(
                AutoEQAuditIssue(
                    code="invalid_frequency",
                    severity="error",
                    message="EQ band frequency must be positive.",
                    details={"label": label, "frequency_hz": band.frequency_hz},
                )
            )
        if abs(band.gain_db) > config.max_abs_gain_db:
            issues.append(
                AutoEQAuditIssue(
                    code="eq_gain_limit_exceeded",
                    severity="error",
                    message="EQ band gain exceeds configured audit limit.",
                    details={"label": label, "gain_db": band.gain_db, "limit_db": config.max_abs_gain_db},
                )
            )
        if band.q <= 0 or band.q > config.max_q:
            issues.append(
                AutoEQAuditIssue(
                    code="eq_q_limit_exceeded",
                    severity="error",
                    message="EQ band Q exceeds configured audit limit.",
                    details={"label": label, "q": band.q, "limit": config.max_q},
                )
            )
        total_positive_gain += max(0.0, float(band.gain_db))

    if total_positive_gain > config.max_total_positive_gain_db:
        issues.append(
            AutoEQAuditIssue(
                code="positive_gain_budget_exceeded",
                severity="warning",
                message="Total positive EQ gain exceeds budget; add global output trim.",
                details={
                    "total_positive_gain_db": round(total_positive_gain, 3),
                    "budget_db": config.max_total_positive_gain_db,
                    "recommended_global_trim_db": round(config.max_total_positive_gain_db - total_positive_gain, 3),
                },
            )
        )

    return AutoEQAuditResult(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        metadata={
            "band_count": len(peq_bands),
            "total_positive_gain_db": round(total_positive_gain, 3),
            "schema_version": SCHEMA_VERSION,
        },
    )


def audit_biquads(
    biquads: Sequence[BiquadCoefficients],
    config: AutoEQAuditConfig = AutoEQAuditConfig(),
) -> AutoEQAuditResult:
    """Validate stability and metadata limits for an IIR biquad package."""

    issues: List[AutoEQAuditIssue] = []
    max_radius = 0.0
    for index, biquad in enumerate(biquads):
        label = biquad.label or f"biquad_{index + 1}"
        try:
            poles = biquad_poles(biquad)
        except ValueError as exc:
            issues.append(
                AutoEQAuditIssue(
                    code="invalid_biquad",
                    severity="error",
                    message=str(exc),
                    details={"label": label},
                )
            )
            continue

        radii = np.abs(poles)
        max_radius = max(max_radius, float(np.max(radii)) if radii.size else 0.0)
        if np.any(radii >= config.max_pole_radius):
            issues.append(
                AutoEQAuditIssue(
                    code="unstable_biquad",
                    severity="error",
                    message="Biquad poles must stay inside the unit circle.",
                    details={
                        "label": label,
                        "max_pole_radius": float(np.max(radii)),
                        "limit": config.max_pole_radius,
                    },
                )
            )

        if abs(biquad.gain_db) > config.max_abs_gain_db:
            issues.append(
                AutoEQAuditIssue(
                    code="biquad_gain_limit_exceeded",
                    severity="error",
                    message="Biquad metadata gain exceeds configured audit limit.",
                    details={"label": label, "gain_db": biquad.gain_db, "limit_db": config.max_abs_gain_db},
                )
            )

        if biquad.q is not None and (biquad.q <= 0 or biquad.q > config.max_q):
            issues.append(
                AutoEQAuditIssue(
                    code="biquad_q_limit_exceeded",
                    severity="error",
                    message="Biquad metadata Q exceeds configured audit limit.",
                    details={"label": label, "q": biquad.q, "limit": config.max_q},
                )
            )

    return AutoEQAuditResult(
        passed=not any(issue.severity == "error" for issue in issues),
        issues=tuple(issues),
        metadata={"biquad_count": len(biquads), "max_pole_radius": round(max_radius, 6), "schema_version": SCHEMA_VERSION},
    )


def biquad_frequency_response_db(
    biquads: Sequence[BiquadCoefficients],
    frequencies_hz: Sequence[float],
    sample_rate: float,
) -> np.ndarray:
    """Calculate combined biquad magnitude response in dB."""

    freqs = np.asarray(frequencies_hz, dtype=float)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if np.any(freqs < 0) or np.any(freqs > sample_rate / 2):
        raise ValueError("frequencies_hz must be inside [0, Nyquist]")

    omega = 2.0 * np.pi * freqs / sample_rate
    z1 = np.exp(-1j * omega)
    z2 = np.exp(-2j * omega)
    response = np.ones_like(z1, dtype=np.complex128)
    for biquad in biquads:
        bq = biquad.normalized()
        numerator = bq.b0 + bq.b1 * z1 + bq.b2 * z2
        denominator = 1.0 + bq.a1 * z1 + bq.a2 * z2
        denominator = np.where(np.abs(denominator) < MIN_LINEAR_MAGNITUDE, MIN_LINEAR_MAGNITUDE, denominator)
        response *= numerator / denominator
    return np.asarray(lin_to_db(np.abs(response)), dtype=float)


def count_narrow_peaks(
    response_db: Sequence[float],
    threshold_db: float = AutoEQAuditConfig.narrow_peak_threshold_db,
    prominence_db: float = AutoEQAuditConfig.narrow_peak_prominence_db,
) -> int:
    """Count obvious narrow positive peaks in a response curve."""

    values = np.asarray(response_db, dtype=float)
    if values.size < 3:
        return 0
    count = 0
    for idx in range(1, values.size - 1):
        peak = values[idx]
        if peak <= threshold_db:
            continue
        shoulder = max(values[idx - 1], values[idx + 1])
        if peak - shoulder >= prominence_db:
            count += 1
    return count


def audit_frequency_response(
    response_db: Sequence[float],
    config: AutoEQAuditConfig = AutoEQAuditConfig(),
) -> AutoEQAuditResult:
    """Validate that a response curve has no obvious narrow high-gain needles."""

    peak_count = count_narrow_peaks(
        response_db,
        threshold_db=config.narrow_peak_threshold_db,
        prominence_db=config.narrow_peak_prominence_db,
    )
    issues: List[AutoEQAuditIssue] = []
    if peak_count > config.max_narrow_peaks:
        issues.append(
            AutoEQAuditIssue(
                code="narrow_response_peaks",
                severity="error",
                message="AutoEQ response has narrow high-gain peaks.",
                details={"peak_count": peak_count, "limit": config.max_narrow_peaks},
            )
        )
    return AutoEQAuditResult(
        passed=not issues,
        issues=tuple(issues),
        metadata={"narrow_peak_count": peak_count, "schema_version": SCHEMA_VERSION},
    )


def normalize_dataset_id(identifier: str) -> str:
    """Normalize a measurement/file/serial identifier for split leakage checks."""

    text = str(identifier).strip().lower()
    if not text:
        return ""
    path_name = Path(text).name
    return Path(path_name).stem


def validate_dataset_splits(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
) -> AutoEQAuditResult:
    """Ensure train and held-out AutoEQ measurement sets do not overlap."""

    train = {normalize_dataset_id(item) for item in train_ids if normalize_dataset_id(item)}
    test = {normalize_dataset_id(item) for item in test_ids if normalize_dataset_id(item)}
    overlap = sorted(train.intersection(test))
    issues: List[AutoEQAuditIssue] = []
    if overlap:
        issues.append(
            AutoEQAuditIssue(
                code="dataset_split_leakage",
                severity="error",
                message="AutoEQ train/test split contains overlapping identifiers.",
                details={"overlap": overlap},
            )
        )
    return AutoEQAuditResult(
        passed=not issues,
        issues=tuple(issues),
        metadata={"train_count": len(train), "test_count": len(test), "schema_version": SCHEMA_VERSION},
    )


def true_peak_dbfs(signal: Sequence[float] | np.ndarray, sample_rate: float, oversample: int = 4) -> float:
    """Approximate true peak in dBFS using lightweight linear oversampling."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    samples = np.asarray(signal, dtype=float)
    if samples.size == 0:
        return float("-inf")

    peak = float(np.max(np.abs(samples)))
    if oversample > 1 and samples.ndim == 1 and samples.size > 1:
        original_x = np.arange(samples.size, dtype=float)
        interp_x = np.linspace(0.0, float(samples.size - 1), (samples.size - 1) * int(oversample) + 1)
        interp = np.interp(interp_x, original_x, samples)
        peak = max(peak, float(np.max(np.abs(interp))))

    if peak <= 0.0:
        return float("-inf")
    return float(lin_to_db(peak))


def normalize_to_true_peak_headroom(
    signal: Sequence[float] | np.ndarray,
    sample_rate: float,
    margin_db: float = AutoEQAuditConfig.true_peak_margin_db,
    oversample: int = 4,
) -> Tuple[np.ndarray, HeadroomAdjustment]:
    """Apply global negative trim if a signal exceeds the true-peak headroom margin."""

    samples = np.asarray(signal, dtype=float)
    before = true_peak_dbfs(samples, sample_rate, oversample=oversample)
    target_peak = -abs(float(margin_db))
    applied_gain = 0.0
    output = np.array(samples, dtype=float, copy=True)
    if np.isfinite(before) and before > target_peak:
        applied_gain = target_peak - before
        output *= float(db_to_lin(applied_gain))
    after = true_peak_dbfs(output, sample_rate, oversample=oversample)
    return output, HeadroomAdjustment(
        true_peak_before_dbfs=before,
        true_peak_after_dbfs=after,
        applied_gain_db=applied_gain,
        margin_db=abs(float(margin_db)),
    )


def validate_export_reproducibility(
    reference_response_db: Sequence[float],
    exported_response_db: Sequence[float],
    tolerance_db: float = AutoEQAuditConfig.export_tolerance_db,
) -> AutoEQAuditResult:
    """Validate that two exported filter responses match within tolerance."""

    reference = np.asarray(reference_response_db, dtype=float)
    exported = np.asarray(exported_response_db, dtype=float)
    issues: List[AutoEQAuditIssue] = []
    if reference.shape != exported.shape:
        issues.append(
            AutoEQAuditIssue(
                code="export_response_shape_mismatch",
                severity="error",
                message="Reference and exported responses must have the same shape.",
                details={"reference_shape": reference.shape, "exported_shape": exported.shape},
            )
        )
        return AutoEQAuditResult(passed=False, issues=tuple(issues), metadata={"schema_version": SCHEMA_VERSION})

    max_delta = float(np.max(np.abs(reference - exported))) if reference.size else 0.0
    if max_delta > tolerance_db:
        issues.append(
            AutoEQAuditIssue(
                code="export_response_delta_exceeded",
                severity="error",
                message="Exported filter response does not reproduce the reference response.",
                details={"max_delta_db": round(max_delta, 6), "tolerance_db": tolerance_db},
            )
        )

    return AutoEQAuditResult(
        passed=not issues,
        issues=tuple(issues),
        metadata={"max_delta_db": round(max_delta, 6), "tolerance_db": tolerance_db, "schema_version": SCHEMA_VERSION},
    )


def build_autoeq_audit_report(
    *,
    phase_contract: Optional[PhaseContract] = None,
    peq_bands: Sequence[ParametricEQBand] = (),
    biquads: Sequence[BiquadCoefficients] = (),
    response_db: Sequence[float] = (),
    train_ids: Iterable[str] = (),
    test_ids: Iterable[str] = (),
    config: AutoEQAuditConfig = AutoEQAuditConfig(),
) -> Dict[str, Any]:
    """Build a JSON-safe aggregate AutoEQ audit report."""

    results: Dict[str, AutoEQAuditResult] = {}
    if phase_contract is not None:
        results["phase_contract"] = validate_phase_contract(phase_contract)
    if peq_bands:
        results["peq_limits"] = audit_peq_bands(peq_bands, config=config)
    if biquads:
        results["biquad_stability"] = audit_biquads(biquads, config=config)
    response_tuple = tuple(response_db)
    if response_tuple:
        results["frequency_response"] = audit_frequency_response(response_tuple, config=config)
    train_list = tuple(train_ids)
    test_list = tuple(test_ids)
    if train_list or test_list:
        results["dataset_splits"] = validate_dataset_splits(train_list, test_list)

    all_issues = [issue for result in results.values() for issue in result.issues]
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": not any(issue.severity == "error" for issue in all_issues),
        "results": {name: result.to_dict() for name, result in results.items()},
        "issue_count": len(all_issues),
        "error_count": sum(1 for issue in all_issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in all_issues if issue.severity == "warning"),
    }
