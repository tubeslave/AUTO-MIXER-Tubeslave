"""Offline ABX/loudness gate for Automixer evaluation artifacts.

This tool evaluates local JSON artifacts only. It does not import mixer
transports and it does not apply console changes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VARIANT_ALIASES = {
    "a": "A",
    "before": "A",
    "original": "A",
    "reference": "A",
    "baseline": "A",
    "b": "B",
    "after": "B",
    "candidate": "B",
    "processed": "B",
    "test": "B",
}


@dataclass(frozen=True)
class ABXGateConfig:
    max_lufs_delta: float = 0.5
    peak_ceiling_dbfs: float = -0.1
    min_trials_for_preference: int = 30
    preference_alpha: float = 0.05
    require_seed: bool = True
    require_checksums: bool = True
    require_preference: bool = False
    expected_preferred_variant: Optional[str] = None
    require_safety_audits: bool = False


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def _normalize_variant(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return VARIANT_ALIASES.get(text)


def _first_present(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _clip_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = payload.get("clips")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, Mapping)]
    raw = payload.get("clip_metrics")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, Mapping)]
    return []


def _trial_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = payload.get("trials")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, Mapping)]
    raw = payload.get("abx_trials")
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, Mapping)]
    return []


def _audit_rows(payload: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    for key in ("safety_audits", "audit_reports", "audits"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [row for row in raw if isinstance(row, Mapping)]
        if isinstance(raw, Mapping):
            if any(status_key in raw for status_key in ("passed", "overall_pass", "technical_pass")):
                return [raw]
            rows = []
            for name, value in raw.items():
                if isinstance(value, Mapping):
                    merged = {"name": str(name)}
                    merged.update(value)
                    rows.append(merged)
            return rows
    return []


def _seed_present(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    if payload.get("seed") is not None or payload.get("random_seed") is not None:
        return True
    if not rows:
        return False
    return all(row.get("seed") is not None or row.get("random_seed") is not None for row in rows)


def _checksum_present(row: Mapping[str, Any]) -> bool:
    return any(row.get(name) for name in ("checksum", "sha256", "audio_sha256", "file_sha256"))


def _checksum_values(row: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for name in ("checksum", "sha256", "audio_sha256", "file_sha256"):
        value = row.get(name)
        if value:
            values.append(str(value))
    return values


def _audit_target_checksum(row: Mapping[str, Any]) -> Optional[str]:
    for name in (
        "target_render_sha256",
        "candidate_sha256",
        "candidate_audio_sha256",
        "checked_audio_sha256",
        "render_sha256",
        "audio_sha256",
    ):
        value = row.get(name)
        if value:
            return str(value)

    for name in ("target", "candidate", "render", "subject"):
        value = row.get(name)
        if isinstance(value, Mapping):
            checksums = _checksum_values(value)
            if checksums:
                return checksums[0]
    return None


def _candidate_b_checksums(clips: Sequence[Mapping[str, Any]]) -> List[str]:
    values: List[str] = []
    for row in clips:
        variant = _normalize_variant(_first_present(row, ("variant", "side", "label", "version")))
        if variant == "B":
            values.extend(_checksum_values(row))
    return values


def _audit_status(row: Mapping[str, Any]) -> Optional[bool]:
    for name in ("passed", "overall_pass", "technical_pass"):
        if name in row:
            value = row[name]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"pass", "passed", "true", "ok"}:
                    return True
                if normalized in {"fail", "failed", "false", "blocked", "error"}:
                    return False
            return None
    return None


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p_hat + z2 / (2.0 * trials)) / denominator
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * trials)) / trials) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def two_sided_binomial_p(successes: int, trials: int, probability: float = 0.5) -> float:
    if trials <= 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be within [0, trials]")
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("probability must be between 0 and 1")

    def pmf(k: int) -> float:
        return math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * math.log(probability)
            + (trials - k) * math.log(1.0 - probability)
        )

    observed = pmf(successes)
    total = 0.0
    for k in range(trials + 1):
        candidate = pmf(k)
        if candidate <= observed + 1e-15:
            total += candidate
    return min(1.0, total)


def _variant_metrics(
    clips: Sequence[Mapping[str, Any]],
    failures: List[Dict[str, str]],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    lufs_by_variant: Dict[str, List[float]] = {"A": [], "B": []}
    peaks_by_variant: Dict[str, List[float]] = {"A": [], "B": []}
    for index, row in enumerate(clips):
        variant = _normalize_variant(_first_present(row, ("variant", "side", "label", "version")))
        if variant not in lufs_by_variant:
            failures.append({"code": "unknown_clip_variant", "detail": f"clips[{index}] has no A/B variant"})
            continue

        lufs = _as_float(_first_present(row, ("lufs", "integrated_lufs", "lufs_integrated", "mean_lufs")))
        peak = _as_float(_first_present(row, ("peak_dbfs", "true_peak_dbfs", "true_peak_dbtp", "peak")))
        if lufs is None:
            failures.append({"code": "missing_lufs", "detail": f"clips[{index}] missing numeric LUFS"})
        else:
            lufs_by_variant[variant].append(lufs)
        if peak is None:
            failures.append({"code": "missing_peak", "detail": f"clips[{index}] missing numeric peak"})
        else:
            peaks_by_variant[variant].append(peak)
    return lufs_by_variant, peaks_by_variant


def _preference_stats(
    payload: Mapping[str, Any],
    config: ABXGateConfig,
    warnings: List[Dict[str, str]],
) -> Dict[str, Any]:
    trials = _trial_rows(payload)
    choices: List[str] = []
    for row in trials:
        choice = _normalize_variant(
            _first_present(row, ("preferred_variant", "choice_variant", "choice", "winner", "selected_variant"))
        )
        if choice in {"A", "B"}:
            choices.append(choice)

    n = len(choices)
    if n == 0:
        warnings.append({"code": "no_abx_trials", "detail": "No usable ABX trial choices were found"})
        return {
            "n": 0,
            "observed_preferred_variant": None,
            "preferred_count": 0,
            "preferred_rate": None,
            "binomial_p": None,
            "wilson_ci_95": [None, None],
            "preference_claim_allowed": False,
        }

    expected = config.expected_preferred_variant or payload.get("expected_preferred_variant")
    expected_variant = _normalize_variant(expected)
    counts = {"A": choices.count("A"), "B": choices.count("B")}
    if expected_variant is None:
        if counts["A"] == counts["B"]:
            observed_variant = None
            preferred_count = counts["A"]
            warnings.append({"code": "abx_tie", "detail": "A/B choices are tied; no preference can be claimed"})
        else:
            observed_variant = "A" if counts["A"] > counts["B"] else "B"
            preferred_count = counts[observed_variant]
    else:
        observed_variant = expected_variant
        preferred_count = counts[expected_variant]

    preferred_rate = preferred_count / n
    p_value = two_sided_binomial_p(preferred_count, n)
    lower, upper = wilson_interval(preferred_count, n)
    preference_allowed = (
        observed_variant is not None
        and n >= int(config.min_trials_for_preference)
        and preferred_rate > 0.5
        and p_value < float(config.preference_alpha)
    )
    return {
        "n": n,
        "observed_preferred_variant": observed_variant,
        "preferred_count": preferred_count,
        "preferred_rate": _round(preferred_rate),
        "binomial_p": _round(p_value),
        "wilson_ci_95": [_round(lower), _round(upper)],
        "preference_claim_allowed": preference_allowed,
        "choice_counts": counts,
    }


def _safety_audit_summary(
    payload: Mapping[str, Any],
    config: ABXGateConfig,
    candidate_b_checksums: Sequence[str],
    failures: List[Dict[str, str]],
) -> Dict[str, Any]:
    audits = _audit_rows(payload)
    missing_status: List[str] = []
    failed: List[str] = []
    passed: List[str] = []
    missing_provenance: List[str] = []
    mismatched_provenance: List[str] = []
    with_provenance: List[str] = []
    for index, row in enumerate(audits):
        name = str(row.get("name") or row.get("tool") or row.get("schema_version") or f"audit[{index}]")
        status = _audit_status(row)
        if status is True:
            passed.append(name)
            if config.require_safety_audits:
                target_checksum = _audit_target_checksum(row)
                if not target_checksum:
                    missing_provenance.append(name)
                elif not candidate_b_checksums:
                    mismatched_provenance.append(name)
                elif target_checksum not in set(candidate_b_checksums):
                    mismatched_provenance.append(name)
                else:
                    with_provenance.append(name)
        elif status is False:
            failed.append(name)
        else:
            missing_status.append(name)

    if config.require_safety_audits and not audits:
        failures.append(
            {
                "code": "missing_safety_audit",
                "detail": "Artifact must include at least one safety audit report",
            }
        )
    for name in failed:
        failures.append({"code": "safety_audit_failed", "detail": f"Attached safety audit failed: {name}"})
    for name in missing_status:
        failures.append(
            {
                "code": "safety_audit_missing_status",
                "detail": f"Attached safety audit lacks pass/fail status: {name}",
            }
        )

    for name in missing_provenance:
        failures.append(
            {
                "code": "safety_audit_missing_provenance",
                "detail": f"Required safety audit lacks target render checksum: {name}",
            }
        )
    for name in mismatched_provenance:
        failures.append(
            {
                "code": "safety_audit_render_mismatch",
                "detail": f"Required safety audit target checksum does not match ABX candidate: {name}",
            }
        )

    return {
        "required": bool(config.require_safety_audits),
        "count": len(audits),
        "passed": passed,
        "failed": failed,
        "missing_status": missing_status,
        "with_provenance": with_provenance,
        "missing_provenance": missing_provenance,
        "mismatched_provenance": mismatched_provenance,
    }


def evaluate_abx_artifact(payload: Mapping[str, Any], config: ABXGateConfig = ABXGateConfig()) -> Dict[str, Any]:
    failures: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    clips = _clip_rows(payload)
    trials = _trial_rows(payload)
    all_rows: List[Mapping[str, Any]] = list(clips) + list(trials)

    if config.require_seed and not _seed_present(payload, all_rows):
        failures.append({"code": "missing_seed", "detail": "Artifact must carry a top-level or per-row seed"})

    checksum_count = sum(1 for row in clips if _checksum_present(row))
    if config.require_checksums and clips and checksum_count != len(clips):
        failures.append({"code": "missing_clip_checksum", "detail": "Each clip must carry checksum evidence"})
    if not clips:
        failures.append({"code": "missing_clips", "detail": "Artifact must include clips or clip_metrics"})

    lufs_by_variant, peaks_by_variant = _variant_metrics(clips, failures)
    mean_lufs = {variant: _mean(values) for variant, values in lufs_by_variant.items()}
    max_peak = {variant: max(values) if values else None for variant, values in peaks_by_variant.items()}

    if mean_lufs["A"] is None or mean_lufs["B"] is None:
        failures.append({"code": "missing_variant_lufs", "detail": "Both A and B require LUFS metrics"})
        lufs_delta = None
    else:
        lufs_delta = abs(float(mean_lufs["A"]) - float(mean_lufs["B"]))
        if lufs_delta > float(config.max_lufs_delta):
            failures.append(
                {
                    "code": "lufs_mismatch",
                    "detail": f"A/B LUFS delta {lufs_delta:.3f} exceeds {config.max_lufs_delta:.3f}",
                }
            )

    for variant, peak in max_peak.items():
        if peak is None:
            failures.append({"code": "missing_variant_peak", "detail": f"Variant {variant} requires peak metrics"})
        elif peak > float(config.peak_ceiling_dbfs):
            failures.append(
                {
                    "code": "peak_ceiling_exceeded",
                    "detail": f"Variant {variant} peak {peak:.3f} exceeds {config.peak_ceiling_dbfs:.3f}",
                }
            )

    preference = _preference_stats(payload, config, warnings)
    safety_audits = _safety_audit_summary(payload, config, _candidate_b_checksums(clips), failures)
    if config.require_preference and not preference["preference_claim_allowed"]:
        failures.append({"code": "preference_gate_failed", "detail": "ABX preference claim did not pass"})

    technical_failure_codes = {
        "missing_seed",
        "missing_clip_checksum",
        "missing_clips",
        "unknown_clip_variant",
        "missing_lufs",
        "missing_peak",
        "missing_variant_lufs",
        "missing_variant_peak",
        "lufs_mismatch",
        "peak_ceiling_exceeded",
        "missing_safety_audit",
        "safety_audit_failed",
        "safety_audit_missing_status",
        "safety_audit_missing_provenance",
        "safety_audit_render_mismatch",
    }
    technical_pass = not any(item["code"] in technical_failure_codes for item in failures)
    overall_pass = technical_pass and (not config.require_preference or preference["preference_claim_allowed"])

    return {
        "schema_version": "automixer-abx-gate-v1",
        "overall_pass": overall_pass,
        "technical_pass": technical_pass,
        "preference_pass": bool(preference["preference_claim_allowed"]),
        "failures": failures,
        "warnings": warnings,
        "config": asdict(config),
        "seed_present": _seed_present(payload, all_rows),
        "checksum_coverage": {
            "required": bool(config.require_checksums),
            "clip_count": len(clips),
            "with_checksum": checksum_count,
        },
        "loudness": {
            "mean_lufs_A": _round(mean_lufs["A"]),
            "mean_lufs_B": _round(mean_lufs["B"]),
            "delta_lu": _round(lufs_delta),
            "max_delta_lu": float(config.max_lufs_delta),
        },
        "peaks": {
            "max_peak_A_dbfs": _round(max_peak["A"]),
            "max_peak_B_dbfs": _round(max_peak["B"]),
            "ceiling_dbfs": float(config.peak_ceiling_dbfs),
        },
        "trials": preference,
        "safety_audits": safety_audits,
        "live_write": False,
        "offline_only": True,
    }


def load_json_artifact(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("ABX artifact must be a JSON object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an offline ABX/loudness artifact")
    parser.add_argument("artifact_json", type=Path)
    parser.add_argument("--out", type=Path, help="Write the gate report JSON to this path")
    parser.add_argument("--max-lufs-delta", type=float, default=ABXGateConfig.max_lufs_delta)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=ABXGateConfig.peak_ceiling_dbfs)
    parser.add_argument("--min-trials", type=int, default=ABXGateConfig.min_trials_for_preference)
    parser.add_argument("--alpha", type=float, default=ABXGateConfig.preference_alpha)
    parser.add_argument("--expected-preferred-variant", choices=("A", "B", "a", "b"))
    parser.add_argument("--allow-missing-seed", action="store_true")
    parser.add_argument("--allow-missing-checksums", action="store_true")
    parser.add_argument("--require-preference", action="store_true")
    parser.add_argument("--require-safety-audit", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ABXGateConfig(
        max_lufs_delta=args.max_lufs_delta,
        peak_ceiling_dbfs=args.peak_ceiling_dbfs,
        min_trials_for_preference=args.min_trials,
        preference_alpha=args.alpha,
        require_seed=not args.allow_missing_seed,
        require_checksums=not args.allow_missing_checksums,
        require_preference=args.require_preference,
        expected_preferred_variant=args.expected_preferred_variant,
        require_safety_audits=args.require_safety_audit,
    )
    report = evaluate_abx_artifact(load_json_artifact(args.artifact_json), config)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
