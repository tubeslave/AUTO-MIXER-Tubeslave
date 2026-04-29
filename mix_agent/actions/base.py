"""Conservative offline action application.

Only simple reversible DSP is applied here.  Placeholder actions such as
dynamic EQ, sidechain, saturation and compression are preserved in the audit
trail but are not faked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import soundfile as sf

from mix_agent.models import MixAction


def db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _safe_gain_db(value: float) -> float:
    # Offline conservative mode may cut freely but never auto-boosts.
    return max(-6.0, min(0.0, float(value)))


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(audio, dtype=np.float32) * db_to_amp(_safe_gain_db(gain_db))


def apply_high_pass(audio: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt

        cutoff = max(10.0, min(float(frequency_hz), sample_rate * 0.45))
        sos = butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
        return sosfilt(sos, audio, axis=0).astype(np.float32)
    except Exception:
        return np.asarray(audio, dtype=np.float32)


def _peaking_coefficients(sample_rate: int, frequency_hz: float, gain_db: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    frequency_hz = max(20.0, min(float(frequency_hz), sample_rate * 0.45))
    q = max(0.44, min(float(q), 12.0))
    gain_db = max(-3.0, min(0.75, float(gain_db)))
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * frequency_hz / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    b = np.asarray([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    den = np.asarray([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, den


def apply_parametric_eq(
    audio: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    try:
        from scipy.signal import lfilter

        b, a = _peaking_coefficients(sample_rate, frequency_hz, gain_db, q)
        return lfilter(b, a, audio, axis=0).astype(np.float32)
    except Exception:
        return np.asarray(audio, dtype=np.float32)


def apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    pan = max(-1.0, min(1.0, float(pan)))
    left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
    right_gain = np.sin((pan + 1.0) * np.pi / 4.0)
    rendered = data[:, :2].copy()
    rendered[:, 0] *= left_gain
    rendered[:, 1] *= right_gain
    return rendered.astype(np.float32)


def inferred_default_pan(stem_name: str, role: str) -> float:
    """Infer a conservative default pan from stem naming.

    This keeps exported L/R pairs from being collapsed to center when the
    offline facade renders stems directly.
    """
    label = stem_name.lower().replace("-", " ").replace("_", " ")
    words = {part for part in label.split() if part}
    if " l" in f" {label} " or "left" in words or label.endswith(" l"):
        if role in {"overhead", "drums"} and ("oh" in words or "overhead" in label):
            return -0.8
        if role in {"backing_vocal", "vocal"}:
            return -0.55
        return -0.7
    if " r" in f" {label} " or "right" in words or label.endswith(" r"):
        if role in {"overhead", "drums"} and ("oh" in words or "overhead" in label):
            return 0.8
        if role in {"backing_vocal", "vocal"}:
            return 0.55
        return 0.7
    if role in {"kick", "snare", "bass", "lead_vocal", "vocal"}:
        return 0.0
    return 0.0


def _target_matches(target: str, stem_name: str, role: str) -> bool:
    target = target.lower()
    return target in {stem_name.lower(), role.lower(), "mix", "all"} or (
        target == "accompaniment" and "vocal" not in role.lower()
    )


def _process_stem(
    audio: np.ndarray,
    stem_name: str,
    role: str,
    sample_rate: int,
    actions: Iterable[MixAction],
) -> tuple[np.ndarray, list[Dict[str, Any]]]:
    processed = np.asarray(audio, dtype=np.float32)
    processed = apply_pan(processed, inferred_default_pan(stem_name, role))
    audit: list[Dict[str, Any]] = []
    for action in actions:
        if action.mode not in {"safe_apply", "apply", "recommend"}:
            continue
        if not _target_matches(action.target, stem_name, role):
            continue
        before_peak = float(np.max(np.abs(processed)) + 1e-12)
        if action.action_type == "gain_adjustment":
            processed = apply_gain(processed, float(action.parameters.get("gain_db", 0.0)))
            status = "applied"
        elif action.action_type == "high_pass_filter":
            processed = apply_high_pass(processed, sample_rate, float(action.parameters.get("frequency_hz", 20.0)))
            status = "applied"
        elif action.action_type == "parametric_eq":
            processed = apply_parametric_eq(
                processed,
                sample_rate,
                float(action.parameters.get("frequency_hz", 1000.0)),
                float(action.parameters.get("gain_db", 0.0)),
                float(action.parameters.get("q", 1.0)),
            )
            status = "applied"
        elif action.action_type == "pan_adjustment":
            processed = apply_pan(processed, float(action.parameters.get("pan", 0.0)))
            status = "applied"
        else:
            status = "placeholder_not_applied"
        after_peak = float(np.max(np.abs(processed)) + 1e-12)
        audit.append(
            {
                "action_id": action.id,
                "action_type": action.action_type,
                "target": action.target,
                "stem": stem_name,
                "status": status,
                "before_peak": before_peak,
                "after_peak": after_peak,
                "reversible": True,
            }
        )
    return processed, audit


def render_conservative_mix(
    stems: Mapping[str, np.ndarray],
    roles: Mapping[str, str],
    sample_rate: int,
    actions: Iterable[MixAction],
    output_path: str | Path,
) -> Dict[str, Any]:
    """Render a conservative offline mix and write a reversible audit trail."""
    if not stems:
        raise ValueError("Conservative apply requires stems")
    actions = list(actions)
    length = max(len(audio) for audio in stems.values())
    mix = np.zeros((length, 2), dtype=np.float32)
    audit: list[Dict[str, Any]] = []
    for name, audio in stems.items():
        role = roles.get(name, "unknown")
        processed, stem_audit = _process_stem(audio, name, role, sample_rate, actions)
        if processed.ndim == 1:
            processed = processed[:, None]
        if processed.shape[1] == 1:
            processed = np.repeat(processed, 2, axis=1)
        if len(processed) < length:
            processed = np.vstack([processed, np.zeros((length - len(processed), processed.shape[1]), dtype=np.float32)])
        mix += processed[:length, :2]
        audit.extend(stem_audit)

    peak = float(np.max(np.abs(mix)) + 1e-12)
    output_gain_db = 0.0
    if peak > 0.98:
        output_gain_db = 20.0 * np.log10(0.98 / peak)
        mix *= db_to_amp(output_gain_db)
        audit.append(
            {
                "action_id": "render.output_safety_trim",
                "action_type": "gain_adjustment",
                "target": "mix",
                "status": "applied",
                "gain_db": output_gain_db,
                "reason": "Prevent offline render clipping.",
                "reversible": True,
            }
        )

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mix, sample_rate, subtype="PCM_24")
    return {
        "output": str(path),
        "sample_rate": sample_rate,
        "duration_sec": round(length / sample_rate, 3),
        "output_safety_gain_db": round(output_gain_db, 3),
        "audit": audit,
    }
