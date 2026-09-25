"""Technical export validation for STUDIO mastering deliveries.

This layer validates what was actually written/decoded, not only the in-memory
mastering candidate. Passing it never promotes an audio baseline or replaces
human listening.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np


_ABSOLUTE_EXPORT_TRUE_PEAK_CEILING_DBTP = -1.0


@dataclass(frozen=True)
class MasterExportValidationPolicy:
    wav_true_peak_tolerance_db: float = 0.05
    lossy_true_peak_tolerance_db: float = 0.35
    loudness_tolerance_lu: float = 0.60
    required_formats: tuple[str, ...] = ("wav", "mp3_decoded")

    def validate(self) -> None:
        if not 0.0 <= float(self.wav_true_peak_tolerance_db) <= 0.5:
            raise ValueError("wav_true_peak_tolerance_db must be in [0, 0.5]")
        if not 0.0 <= float(self.lossy_true_peak_tolerance_db) <= 1.0:
            raise ValueError("lossy_true_peak_tolerance_db must be in [0, 1]")
        if not 0.0 < float(self.loudness_tolerance_lu) <= 2.0:
            raise ValueError("loudness_tolerance_lu must be in (0, 2]")
        if not self.required_formats or any(not str(x).strip() for x in self.required_formats):
            raise ValueError("required_formats must contain non-empty names")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def validate_master_exports(
    export_measurements: Mapping[str, Mapping[str, Any]],
    *,
    target_lufs: float,
    requested_ceiling_dbtp: float,
    expected_frames: int,
    export_frames: Mapping[str, int],
    expected_sample_rate: int | None = None,
    export_sample_rates: Mapping[str, int] | None = None,
    policy: MasterExportValidationPolicy | None = None,
) -> dict[str, Any]:
    """Validate encoded exports against the requested mastering target.

    WAV and decoded lossy formats use separate bounded tolerances because
    quantization/codecs can create small true-peak differences. Tolerance may
    relax a stricter requested ceiling slightly, but can never weaken the
    repository-wide -1.0 dBTP export safety ceiling.
    """
    p = policy or MasterExportValidationPolicy()
    p.validate()
    target = _finite_number(target_lufs)
    ceiling = _finite_number(requested_ceiling_dbtp)
    if target is None or not -40.0 <= target <= -5.0:
        raise ValueError("target_lufs must be finite and in [-40, -5]")
    if ceiling is None or not -12.0 <= ceiling <= -1.0:
        raise ValueError("requested_ceiling_dbtp must be finite and in [-12, -1]")
    if (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, (int, np.integer))
        or int(expected_frames) <= 0
    ):
        raise ValueError("expected_frames must be a positive integer")
    if not isinstance(export_measurements, Mapping) or not isinstance(export_frames, Mapping):
        raise TypeError("export measurements and frame counts must be mappings")
    if expected_sample_rate is not None:
        if (
            isinstance(expected_sample_rate, bool)
            or not isinstance(expected_sample_rate, (int, np.integer))
            or int(expected_sample_rate) < 8000
        ):
            raise ValueError("expected_sample_rate must be an integer >= 8000")
        if not isinstance(export_sample_rates, Mapping):
            raise ValueError("export_sample_rates are required with expected_sample_rate")

    failures: list[str] = []
    per_format: dict[str, Any] = {}
    for fmt in p.required_formats:
        metrics = export_measurements.get(fmt)
        if not isinstance(metrics, Mapping):
            failures.append(f"{fmt}_measurements_missing")
            per_format[fmt] = {"status": "missing"}
            continue

        tp = _finite_number(metrics.get("true_peak_dbtp"))
        lufs = _finite_number(metrics.get("integrated_lufs"))
        lufs_method = str(metrics.get("integrated_lufs_method", ""))
        tolerance = (
            float(p.wav_true_peak_tolerance_db)
            if fmt == "wav"
            else float(p.lossy_true_peak_tolerance_db)
        )
        ceiling_limit = min(
            float(ceiling + tolerance),
            _ABSOLUTE_EXPORT_TRUE_PEAK_CEILING_DBTP,
        )

        fmt_failures: list[str] = []
        if tp is None:
            fmt_failures.append("true_peak_unmeasured")
        elif tp > ceiling_limit:
            fmt_failures.append("requested_ceiling_exceeded")

        if lufs is None or lufs_method != "pyloudnorm":
            fmt_failures.append("loudness_unmeasured")
            loudness_error = None
        else:
            loudness_error = float(lufs - target)
            if abs(loudness_error) > float(p.loudness_tolerance_lu):
                fmt_failures.append("loudness_target_missed")

        frames = export_frames.get(fmt)
        if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)):
            fmt_failures.append("frame_count_unmeasured")
        elif int(frames) != int(expected_frames):
            fmt_failures.append("export_length_mismatch")

        sample_rate = None
        if expected_sample_rate is not None:
            sample_rate = export_sample_rates.get(fmt)
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
                fmt_failures.append("sample_rate_unmeasured")
            elif int(sample_rate) != int(expected_sample_rate):
                fmt_failures.append("sample_rate_mismatch")

        failures.extend(f"{fmt}_{failure}" for failure in fmt_failures)
        per_format[fmt] = {
            "status": "rejected" if fmt_failures else "pass",
            "failures": fmt_failures,
            "true_peak_dbtp": tp,
            "requested_ceiling_dbtp": float(ceiling),
            "true_peak_tolerance_db": tolerance,
            "true_peak_limit_dbtp": ceiling_limit,
            "absolute_true_peak_ceiling_dbtp": _ABSOLUTE_EXPORT_TRUE_PEAK_CEILING_DBTP,
            "integrated_lufs": lufs,
            "integrated_lufs_method": lufs_method or None,
            "loudness_error_lu": loudness_error,
            "loudness_tolerance_lu": float(p.loudness_tolerance_lu),
            "frames": (
                int(frames)
                if isinstance(frames, (int, np.integer)) and not isinstance(frames, bool)
                else None
            ),
            "expected_frames": int(expected_frames),
            "sample_rate": (
                int(sample_rate)
                if isinstance(sample_rate, (int, np.integer)) and not isinstance(sample_rate, bool)
                else None
            ),
            "expected_sample_rate": (
                int(expected_sample_rate) if expected_sample_rate is not None else None
            ),
        }

    return {
        "schema": "studio-master-export-validation-v1",
        "status": "rejected_export_validation" if failures else "machine_safe_technical",
        "accept_technical_delivery": not failures,
        "failures": failures,
        "formats": per_format,
        "target_lufs": float(target),
        "requested_ceiling_dbtp": float(ceiling),
        "absolute_true_peak_ceiling_dbtp": _ABSOLUTE_EXPORT_TRUE_PEAK_CEILING_DBTP,
        "expected_frames": int(expected_frames),
        "expected_sample_rate": (
            int(expected_sample_rate) if expected_sample_rate is not None else None
        ),
        "policy": asdict(p),
        "baseline_eligible": False,
        "requires_human_listening": True,
        "human_acceptance_inferred": False,
        "note": "Technical export validation only; subjective mastering acceptance remains human.",
    }
