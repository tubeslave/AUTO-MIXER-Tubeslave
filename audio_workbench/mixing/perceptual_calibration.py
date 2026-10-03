"""Calibration evidence for STUDIO Perceptual Critic.

This module does not change Perceptual Critic thresholds and cannot promote an
audio baseline.  It records disagreements between bounded machine gates and
human level-matched listening, and exposes a diagnostic foreground-stability
measurement for future calibration work.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ForegroundStabilityPolicy:
    """Measurement policy, deliberately not an acceptance threshold."""

    window_ms: float = 320.0
    hop_ms: float = 160.0
    active_relative_db: float = -24.0
    min_active_windows: int = 6
    lower_percentile: float = 10.0
    upper_percentile: float = 90.0

    def validate(self, sr: int) -> None:
        if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
            raise ValueError("sample rate must be an integer >= 1000 Hz")
        if not 40 <= float(self.window_ms) <= 2000:
            raise ValueError("window_ms must be in [40, 2000]")
        if not 10 <= float(self.hop_ms) <= float(self.window_ms):
            raise ValueError("hop_ms must be in [10, window_ms]")
        if not -60 <= float(self.active_relative_db) <= -3:
            raise ValueError("active_relative_db must be in [-60, -3]")
        if isinstance(self.min_active_windows, bool) or int(self.min_active_windows) < 3:
            raise ValueError("min_active_windows must be >= 3")
        lo, hi = float(self.lower_percentile), float(self.upper_percentile)
        if not 0 <= lo < hi <= 100:
            raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")


def _mono_finite(x: np.ndarray, name: str) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 2:
        if y.shape[1] not in (1, 2):
            raise ValueError(f"{name} must be mono or stereo")
        y = y.mean(axis=1)
    elif y.ndim != 1:
        raise ValueError(f"{name} must be mono or stereo")
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f"{name} must be finite and non-empty")
    return y


def _window_rms_db(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    if len(x) < window:
        return np.empty(0, dtype=np.float64)
    starts = np.arange(0, len(x) - window + 1, hop, dtype=np.int64)
    # Avoid a giant strided matrix: calibration should remain memory-bounded on
    # full songs.  The explicit loop is deterministic and fast for ~0.16 s hops.
    out = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        frame = x[start:start + window]
        out[i] = 20.0 * np.log10(np.sqrt(np.mean(frame * frame) + 1e-20) + 1e-12)
    return out


def foreground_stability_evidence(
    baseline_foreground: np.ndarray,
    candidate_foreground: np.ndarray,
    sr: int,
    *,
    activity_reference: np.ndarray | None = None,
    policy: ForegroundStabilityPolicy | None = None,
) -> dict[str, Any]:
    """Measure robust foreground level spread on fixed baseline-derived windows.

    The activity mask is derived once from ``activity_reference`` (or the baseline
    foreground) and reused for baseline and candidate.  The candidate receives a
    *single constant gain* to match baseline active-window median level before the
    diagnostic tails are compared.  This prevents a simple fader move from being
    mistaken for improved stability.  The result is calibration evidence only;
    it has no production acceptance threshold or baseline-promotion authority.
    """
    p = policy or ForegroundStabilityPolicy()
    p.validate(sr)
    baseline = _mono_finite(baseline_foreground, "baseline_foreground")
    candidate = _mono_finite(candidate_foreground, "candidate_foreground")
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate foreground shapes differ")
    reference = baseline if activity_reference is None else _mono_finite(activity_reference, "activity_reference")
    if reference.shape != baseline.shape:
        raise ValueError("activity_reference shape differs from foreground")
    if np.max(np.abs(baseline)) < 1e-10:
        raise ValueError("baseline_foreground is effectively silent")
    if np.max(np.abs(candidate)) < 1e-10:
        raise ValueError("candidate_foreground is effectively silent")
    if np.max(np.abs(reference)) < 1e-10:
        raise ValueError("activity_reference is effectively silent")

    window = max(1, int(round(float(p.window_ms) * sr / 1000.0)))
    hop = max(1, int(round(float(p.hop_ms) * sr / 1000.0)))
    b_db = _window_rms_db(baseline, window, hop)
    c_db = _window_rms_db(candidate, window, hop)
    r_db = _window_rms_db(reference, window, hop)
    if len(r_db) < int(p.min_active_windows):
        raise ValueError("not enough analysis windows")

    peak = float(np.max(r_db))
    mask = r_db >= peak + float(p.active_relative_db)
    active = int(np.count_nonzero(mask))
    if active < int(p.min_active_windows):
        raise ValueError("not enough active foreground windows")

    b = b_db[mask]
    c = c_db[mask]
    median_match_gain = float(np.median(b) - np.median(c))
    c_matched = c + median_match_gain
    lo, hi = float(p.lower_percentile), float(p.upper_percentile)
    b_lo, b_hi = np.percentile(b, [lo, hi])
    c_lo, c_hi = np.percentile(c_matched, [lo, hi])
    b_med = float(np.median(b))
    c_med = float(np.median(c_matched))
    b_abs = np.abs(b - b_med)
    c_abs = np.abs(c_matched - c_med)
    before_spread = float(b_hi - b_lo)
    after_spread = float(c_hi - c_lo)

    return {
        "schema": "foreground-stability-evidence-v1",
        "role": "diagnostic_calibration_only",
        "window_ms": float(p.window_ms),
        "hop_ms": float(p.hop_ms),
        "activity_reference": "baseline_foreground" if activity_reference is None else "explicit_fixed_reference",
        "active_windows": active,
        "total_windows": int(len(r_db)),
        "active_threshold_dbfs": float(peak + float(p.active_relative_db)),
        "candidate_constant_match_gain_db": median_match_gain,
        "median_level_match_error_db": float(c_med - b_med),
        "spread_before_db": before_spread,
        "spread_after_db": after_spread,
        "spread_improvement_db": float(before_spread - after_spread),
        "p95_absolute_deviation_before_db": float(np.percentile(b_abs, 95)),
        "p95_absolute_deviation_after_db": float(np.percentile(c_abs, 95)),
        "p95_absolute_deviation_improvement_db": float(np.percentile(b_abs, 95) - np.percentile(c_abs, 95)),
        "policy": asdict(p),
        "accept": None,
        "baseline_promotion_allowed": False,
        "requires_human_listening": True,
        "note": "Robust phrase/window level consistency proxy; not intelligibility and not an overall quality score.",
    }


def _human_status(review: dict[str, Any]) -> str:
    raw = str(review.get("human_review", review.get("verdict", ""))).strip().lower()
    aliases = {"accept": "accepted", "approve": "accepted", "approved": "accepted",
               "reject": "rejected", "decline": "rejected"}
    status = aliases.get(raw, raw)
    if status not in {"accepted", "rejected"}:
        raise ValueError("human review must be accepted or rejected")
    return status


def _normalized_tags(review: dict[str, Any]) -> list[str]:
    raw: Iterable[Any] = review.get("observations") or review.get("tags") or []
    if isinstance(raw, str):
        raw = [raw]
    tags = []
    for value in raw:
        tag = str(value).strip().lower().replace(" ", "_")
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def classify_human_machine_disagreement(
    critic_result: dict[str, Any],
    human_review: dict[str, Any],
) -> dict[str, Any]:
    """Record calibration disagreement without overriding either safety layer.

    A human preference can reveal that a *target proxy* missed something audible,
    but it cannot erase protected regressions or retroactively make a failed metric
    pass.  Conversely, a machine-safe candidate still cannot be promoted when the
    listener rejects it.  The output is evidence for future metric calibration.
    """
    if not isinstance(critic_result, dict) or not isinstance(human_review, dict):
        raise TypeError("critic_result and human_review must be dictionaries")
    machine = str(critic_result.get("machine_decision", "")).strip()
    if machine not in {"rejected", "pending_human_review", "machine_safe"}:
        raise ValueError("unsupported or missing critic machine_decision")
    target = str(critic_result.get("target", "")).strip()
    if not target:
        raise ValueError("critic target is required")
    human = _human_status(human_review)
    failures = [str(x) for x in (critic_result.get("failures") or [])]
    protected = [str(x) for x in (critic_result.get("protected_regressions") or [])]
    target_only_rejection = machine == "rejected" and bool(failures) and set(failures) == {"target_not_improved"} and not protected

    if human == "accepted" and target_only_rejection:
        classification = "target_proxy_miss_candidate"
        action = "calibrate_additional_target_metric"
    elif human == "accepted" and protected:
        classification = "human_preference_safety_conflict"
        action = "preserve_safety_rejection_and_review_metric_scope"
    elif human == "rejected" and machine == "machine_safe":
        classification = "machine_false_positive_candidate"
        action = "calibrate_missing_protected_or_target_metric"
    elif human == "rejected" and machine == "pending_human_review":
        classification = "uncertainty_resolved_by_human_rejection"
        action = "retain_candidate_rejection"
    elif human == "accepted" and machine in {"machine_safe", "pending_human_review"}:
        classification = "human_machine_agreement_positive"
        action = "record_agreement_without_auto_promotion"
    else:
        classification = "human_machine_agreement_negative"
        action = "record_agreement"

    tags = _normalized_tags(human_review)
    metric_hypotheses = []
    if any(t in tags for t in ("foreground_stability", "phrase_consistency", "stable_mix_position", "does_not_pop_or_disappear")):
        metric_hypotheses.append("foreground_stability")

    return {
        "schema": "perceptual-human-calibration-v1",
        "target": target,
        "machine_decision": machine,
        "human_review": human,
        "classification": classification,
        "calibration_action": action,
        "machine_failures_preserved": failures,
        "protected_regressions_preserved": protected,
        "observations": tags,
        "metric_hypotheses": metric_hypotheses,
        "threshold_update_allowed": False,
        "protected_gate_override_allowed": False,
        "baseline_promotion_allowed": False,
        "requires_human_listening": True,
        "note": "Calibration evidence is descriptive; it never rewrites historical critic results or grants autonomous taste authority.",
    }
