from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from scipy import signal

@dataclass(frozen=True)
class PerceptualSnapshot:
    foreground_db: float
    vocal_intelligibility: float
    punch_db: float
    harshness: float
    density: float
    depth_proxy: float
    width_db: float
    climax_lift_db: float

@dataclass(frozen=True)
class PerceptualAcceptancePolicy:
    """Conservative engineering defaults; calibrate from listening evidence, not taste labels."""
    vocal_intelligibility_min_improvement: float = .02
    harshness_min_improvement: float = .03
    punch_min_improvement_db: float = .25
    climax_min_improvement_db: float = .20
    max_density_delta: float = .15
    max_width_delta_db: float = 1.5
    max_foreground_delta_db: float = 1.0
    max_foreground_delta_for_vocal_db: float = 1.5
    max_harshness_regression: float = .06
    max_intelligibility_regression: float = .04
    max_punch_regression_db: float = .75
    max_climax_regression_db: float = .50
    uncertainty_review_threshold: float = .65

def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype="float64")**2) + 1e-20))

def _band(x, sr, lo, hi):
    hi = min(hi, sr * .47)
    sos = signal.butter(3, [lo, hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x, axis=0)

def _db_ratio(a, b):
    return float(20 * np.log10((_rms(a) + 1e-12) / (_rms(b) + 1e-12)))

def snapshot(mix: np.ndarray, sr: int, vocal: np.ndarray | None = None, drums: np.ndarray | None = None,
             early_room: np.ndarray | None = None, section_rms_db: list[float] | None = None) -> PerceptualSnapshot:
    if mix.ndim == 1:
        mix = np.column_stack([mix, mix])
    mono = mix.mean(1)
    mid = (mix[:, 0] + mix[:, 1]) * .5
    side = (mix[:, 0] - mix[:, 1]) * .5

    # Harshness is a bounded spectral prominence proxy, not a taste score.
    harsh = _rms(_band(mono, sr, 2500, 6500)) / (_rms(_band(mono, sr, 250, 2200)) + 1e-12)
    harsh = float(np.clip(harsh * 2.2, 0, 1))

    # Density: fraction of short windows close to the upper RMS envelope.
    hop = max(1, int(.05 * sr))
    n = len(mono) // hop
    if n:
        q = mono[:n * hop].reshape(n, hop)
        db = 20 * np.log10(np.sqrt(np.mean(q.astype("float64")**2, axis=1)) + 1e-12)
        density = float(np.mean(db > np.percentile(db, 80) - 6))
    else:
        density = 0.

    if vocal is not None:
        if vocal.ndim > 1:
            vocal = vocal.mean(1)
        nv = min(len(vocal), len(mono))
        v = vocal[:nv]
        m = mono[:nv]
        fg = _db_ratio(v, m - v)
        vb = _rms(_band(v, sr, 1200, 4500))
        mb = _rms(_band(m - v, sr, 1200, 4500))
        intellig = float(np.clip((20 * np.log10((vb + 1e-12) / (mb + 1e-12)) + 18) / 24, 0, 1))
    else:
        fg = 0.
        intellig = .5

    if drums is not None:
        if drums.ndim > 1:
            drums = drums.mean(1)
        d = drums[:len(mono)]
        low = _band(d, sr, 45, 180)
        env = np.abs(signal.hilbert(low))
        punch = float(20 * np.log10((np.percentile(env, 99) + 1e-12) / (np.percentile(env, 75) + 1e-12)))
    else:
        punch = 0.

    if early_room is not None:
        er = early_room[:len(mix)]
        depth = float(np.clip((_rms(er) / (_rms(mix) + 1e-12)) * 8, 0, 1))
    else:
        depth = .5

    width = _db_ratio(side, mid)
    climax = 0.
    if section_rms_db and len(section_rms_db) >= 3:
        a = np.asarray(section_rms_db, float)
        climax = float(np.percentile(a, 90) - np.median(a))

    return PerceptualSnapshot(fg, intellig, punch, harsh, density, depth, width, climax)

def diagnose(s: PerceptualSnapshot) -> list[dict]:
    """Return hypotheses, never an overall quality score."""
    h = []
    if s.vocal_intelligibility < .42:
        h.append({"target": "vocal_intelligibility", "direction": "increase", "confidence": .78,
                  "bounded_action": "reduce competing 1.2-4.5 kHz or ride vocal <=0.7 dB"})
    if s.harshness > .72:
        h.append({"target": "harshness", "direction": "decrease", "confidence": .72,
                  "bounded_action": "dynamic 2.5-6.5 kHz control <=1.0 dB"})
    if s.punch_db < 5.0:
        h.append({"target": "punch", "direction": "increase", "confidence": .68,
                  "bounded_action": "transient/density adjustment on drums <=0.8 dB"})
    if s.climax_lift_db < 1.0:
        h.append({"target": "climax", "direction": "increase", "confidence": .60,
                  "bounded_action": "section orchestration/width/FX move, not broadband gain first"})
    return sorted(h, key=lambda x: x["confidence"], reverse=True)

def compare(before: PerceptualSnapshot, after: PerceptualSnapshot, target: str) -> dict:
    b = asdict(before)
    a = asdict(after)
    key = {"vocal_intelligibility": "vocal_intelligibility", "harshness": "harshness",
           "punch": "punch_db", "climax": "climax_lift_db"}[target]
    return {"target": target, "before": b[key], "after": a[key], "delta": a[key] - b[key],
            "collateral": {"density": a["density"] - b["density"], "width_db": a["width_db"] - b["width_db"],
                           "foreground_db": a["foreground_db"] - b["foreground_db"]}}

def _bounded_proximity(value: float, limit: float) -> float:
    if limit <= 0:
        return 1.0
    return float(np.clip(max(0.0, value) / limit, 0.0, 1.0))

def accept_candidate(before: PerceptualSnapshot, after: PerceptualSnapshot, target: str,
                     policy: PerceptualAcceptancePolicy | None = None) -> dict:
    """Evaluate one bounded perceptual hypothesis with evidence and uncertainty.

    ``machine_decision`` is an engineering gate, not subjective mix acceptance. Any
    perceptual change still requires human listening before it becomes a baseline.
    """
    p = policy or PerceptualAcceptancePolicy()
    b = asdict(before)
    a = asdict(after)
    target_spec = {
        "vocal_intelligibility": ("vocal_intelligibility", 1, p.vocal_intelligibility_min_improvement),
        "harshness": ("harshness", -1, p.harshness_min_improvement),
        "punch": ("punch_db", 1, p.punch_min_improvement_db),
        "climax": ("climax_lift_db", 1, p.climax_min_improvement_db),
    }
    if target not in target_spec:
        raise ValueError(f"Unsupported perceptual target: {target}")

    key, direction, min_improvement = target_spec[target]
    raw_delta = float(a[key] - b[key])
    improvement = float(raw_delta * direction)
    failures: list[str] = []
    evidence: list[dict] = []
    uncertainty_components: dict[str, float] = {}

    target_passed = improvement >= min_improvement
    if not target_passed:
        failures.append("target_not_improved")
    evidence.append({
        "metric": key,
        "role": "target",
        "before": float(b[key]),
        "after": float(a[key]),
        "delta": raw_delta,
        "improvement": improvement,
        "required_improvement": float(min_improvement),
        "passed": target_passed,
    })
    target_excess = max(0.0, improvement - min_improvement)
    uncertainty_components[f"target:{key}"] = float(
        np.clip(1.0 - target_excess / max(min_improvement, 1e-12), 0.0, 1.0)
    )

    density_delta = float(a["density"] - b["density"])
    width_delta = float(a["width_db"] - b["width_db"])
    foreground_delta = float(a["foreground_db"] - b["foreground_db"])
    foreground_limit = (p.max_foreground_delta_for_vocal_db
                        if target == "vocal_intelligibility" else p.max_foreground_delta_db)

    symmetric = (
        ("density", density_delta, p.max_density_delta, "density_regression"),
        ("width_db", width_delta, p.max_width_delta_db, "width_regression"),
        ("foreground_db", foreground_delta, foreground_limit, "foreground_regression"),
    )
    for metric, delta, limit, failure in symmetric:
        passed = abs(delta) <= limit
        if not passed:
            failures.append(failure)
        evidence.append({
            "metric": metric,
            "role": "protected",
            "before": float(b[metric]),
            "after": float(a[metric]),
            "delta": float(delta),
            "limit": float(limit),
            "mode": "absolute_delta",
            "passed": passed,
        })
        uncertainty_components[f"protected:{metric}"] = _bounded_proximity(abs(delta), limit)

    one_sided = (
        ("harshness", float(a["harshness"] - b["harshness"]), p.max_harshness_regression,
         "harshness_regression", target == "harshness"),
        ("vocal_intelligibility", float(b["vocal_intelligibility"] - a["vocal_intelligibility"]),
         p.max_intelligibility_regression, "intelligibility_regression", target == "vocal_intelligibility"),
        ("punch_db", float(b["punch_db"] - a["punch_db"]), p.max_punch_regression_db,
         "punch_regression", target == "punch"),
        ("climax_lift_db", float(b["climax_lift_db"] - a["climax_lift_db"]), p.max_climax_regression_db,
         "climax_regression", target == "climax"),
    )
    for metric, regression, limit, failure, is_target in one_sided:
        if is_target:
            continue
        passed = regression <= limit
        if not passed:
            failures.append(failure)
        delta = float(a[metric] - b[metric])
        evidence.append({
            "metric": metric,
            "role": "protected",
            "before": float(b[metric]),
            "after": float(a[metric]),
            "delta": delta,
            "regression": max(0.0, regression),
            "limit": float(limit),
            "mode": "one_sided_regression",
            "passed": passed,
        })
        uncertainty_components[f"protected:{metric}"] = _bounded_proximity(regression, limit)

    # A failed check is a rejection, not "uncertainty". Uncertainty only separates
    # comfortably passing evidence from evidence that is too close to a boundary.
    uncertainty_score = max(uncertainty_components.values(), default=0.0)
    uncertainty_reasons = sorted(
        key_name for key_name, value in uncertainty_components.items()
        if value >= p.uncertainty_review_threshold
    )
    protected_regressions = [f for f in failures if f.endswith("_regression")]

    if failures:
        machine_decision = "rejected"
    elif uncertainty_score >= p.uncertainty_review_threshold:
        machine_decision = "pending_human_review"
    else:
        machine_decision = "machine_safe"

    collateral = {
        "density": density_delta,
        "width_db": width_delta,
        "foreground_db": foreground_delta,
        "harshness": float(a["harshness"] - b["harshness"]),
        "vocal_intelligibility": float(a["vocal_intelligibility"] - b["vocal_intelligibility"]),
        "punch_db": float(a["punch_db"] - b["punch_db"]),
        "climax_lift_db": float(a["climax_lift_db"] - b["climax_lift_db"]),
    }
    return {
        # Backward-compatible conservative gate: pending evidence is not accepted.
        "accept": machine_decision == "machine_safe",
        "machine_decision": machine_decision,
        "requires_human_listening": True,
        "failures": failures,
        "protected_regressions": protected_regressions,
        "target": target,
        "target_before": float(b[key]),
        "target_after": float(a[key]),
        "target_improvement": improvement,
        "evidence": evidence,
        "uncertainty": {
            "score": uncertainty_score,
            "review_threshold": float(p.uncertainty_review_threshold),
            "reasons": uncertainty_reasons,
            "components": uncertainty_components,
        },
        "collateral": collateral,
    }
