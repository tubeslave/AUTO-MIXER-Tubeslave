"""Snare top/bottom balance correction for the Ayaic offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import db_to_amp, ensure_stereo


SNARE_TOP_BOTTOM_BALANCE_STAGE = "snare_top_bottom_balance"
TARGET_BOTTOM_RELATIVE_DB = -12.0
TARGET_BOTTOM_MIN_DB = -14.0
TARGET_BOTTOM_MAX_DB = -10.0
MAX_BOTTOM_TRIM_DB = -24.0

AudioFn = Callable[[Any], np.ndarray]
LufsFn = Callable[[np.ndarray, int], float]
PeakFn = Callable[[np.ndarray], float]


def apply_snare_top_bottom_balance(
    stems: list[Any],
    *,
    primary_audio_fn: AudioFn | None = None,
    lufs_fn: LufsFn | None = None,
    peak_fn: PeakFn | None = None,
    report_only: bool = False,
    target_relative_db: float = TARGET_BOTTOM_RELATIVE_DB,
    target_min_db: float = TARGET_BOTTOM_MIN_DB,
    target_max_db: float = TARGET_BOTTOM_MAX_DB,
) -> dict[str, Any]:
    """Trim snare bottom so it sits below snare top by a musical close-mic ratio.

    The stage only attenuates bottom microphones. It never boosts snare top,
    because boosting the top mic after compression/output balance would raise
    peak risk and can make the snare poke out of the mix.
    """

    pair = _find_snare_pair(stems)
    if pair is None:
        return {
            "enabled": False,
            "applied": False,
            "stage": SNARE_TOP_BOTTOM_BALANCE_STAGE,
            "reason": "snare_top_bottom_pair_not_found",
            "pairs": [],
        }

    top, bottom = pair
    sample_rate = int(top.sample_rate)
    top_primary = _primary_audio(top, primary_audio_fn)
    bottom_primary = _primary_audio(bottom, primary_audio_fn)
    top_primary_lufs = _call_lufs(lufs_fn, top_primary, sample_rate)
    bottom_primary_lufs = _call_lufs(lufs_fn, bottom_primary, sample_rate)
    top_full_lufs = _call_lufs(lufs_fn, top.audio, sample_rate)
    bottom_full_lufs = _call_lufs(lufs_fn, bottom.audio, sample_rate)
    before_relative = bottom_primary_lufs - top_primary_lufs
    target = float(np.clip(target_relative_db, target_min_db, target_max_db))

    requested_trim = target - before_relative
    trim = min(0.0, requested_trim)
    trim = max(MAX_BOTTOM_TRIM_DB, trim)
    applied = bool(abs(trim) >= 0.05 and not report_only)
    if applied:
        bottom.audio = ensure_stereo(bottom.audio) * db_to_amp(trim)
        bottom.track_gain_db += trim

    bottom_primary_after = bottom_primary_lufs + trim
    bottom_full_after = bottom_full_lufs + trim
    after_relative = bottom_primary_after - top_primary_lufs
    full_after_relative = bottom_full_after - top_full_lufs

    report = {
        "stage": SNARE_TOP_BOTTOM_BALANCE_STAGE,
        "top_channel": int(top.channel_id),
        "bottom_channel": int(bottom.channel_id),
        "top_file": Path(top.path).name,
        "bottom_file": Path(bottom.path).name,
        "target_bottom_relative_db": round(target, 3),
        "acceptable_range_db": [round(float(target_min_db), 3), round(float(target_max_db), 3)],
        "top_primary_lufs": round(top_primary_lufs, 3),
        "bottom_primary_lufs_before": round(bottom_primary_lufs, 3),
        "bottom_primary_lufs_after": round(bottom_primary_after, 3),
        "primary_relative_before_db": round(before_relative, 3),
        "primary_relative_after_db": round(after_relative, 3),
        "top_full_lufs": round(top_full_lufs, 3),
        "bottom_full_lufs_before": round(bottom_full_lufs, 3),
        "bottom_full_lufs_after": round(bottom_full_after, 3),
        "full_relative_after_db": round(full_after_relative, 3),
        "requested_trim_db": round(requested_trim, 3),
        "applied_trim_db": round(trim, 3),
        "top_peak_dbfs": round(_call_peak(peak_fn, top.audio), 3),
        "bottom_peak_dbfs_before": round(_call_peak(peak_fn, bottom.audio / db_to_amp(trim) if applied else bottom.audio), 3),
        "bottom_peak_dbfs_after": round(_call_peak(peak_fn, bottom.audio), 3),
        "applied": applied,
        "report_only": bool(report_only),
        "reason": _reason(before_relative, after_relative, trim),
    }
    _append_snare_balance_note(top, bottom, report)

    return {
        "enabled": True,
        "applied": applied,
        "stage": SNARE_TOP_BOTTOM_BALANCE_STAGE,
        "method": "post_balance_snare_bottom_relative_trim",
        "pairs": [report],
        "critic": {
            "decision": "ok" if target_min_db <= after_relative <= target_max_db else "review",
            "primary_relative_after_db": round(after_relative, 3),
            "full_relative_after_db": round(full_after_relative, 3),
        },
    }


def _find_snare_pair(stems: list[Any]) -> tuple[Any, Any] | None:
    top = None
    bottom = None
    for stem in stems:
        role = _snare_role(stem.name)
        if role == "top" and top is None:
            top = stem
        elif role == "bottom" and bottom is None:
            bottom = stem
    if top is None or bottom is None:
        return None
    return top, bottom


def _snare_role(name: str) -> str | None:
    key = " ".join(Path(str(name)).stem.lower().replace("_", " ").replace("-", " ").split())
    if "snare" not in key and not key.startswith("sn "):
        return None
    if any(token in key for token in ("bottom", "bot", "btm", " b", "sn b")):
        return "bottom"
    if any(token in key for token in ("top", " t", "sn t")):
        return "top"
    return None


def _primary_audio(stem: Any, primary_audio_fn: AudioFn | None) -> np.ndarray:
    if primary_audio_fn is None:
        return ensure_stereo(stem.audio)
    audio = ensure_stereo(primary_audio_fn(stem))
    if audio.size == 0:
        return ensure_stereo(stem.audio)
    return audio


def _call_lufs(fn: LufsFn | None, audio: np.ndarray, sample_rate: int) -> float:
    if fn is None:
        return _rms_db(audio)
    value = float(fn(audio, sample_rate))
    return value if np.isfinite(value) else -120.0


def _call_peak(fn: PeakFn | None, audio: np.ndarray) -> float:
    if fn is None:
        arr = np.asarray(audio, dtype=np.float64)
        return 20.0 * float(np.log10(float(np.max(np.abs(arr))) + 1e-12)) if arr.size else -120.0
    value = float(fn(audio))
    return value if np.isfinite(value) else -120.0


def _rms_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return -120.0
    return 20.0 * float(np.log10(float(np.sqrt(np.mean(arr * arr))) + 1e-12))


def _reason(before_relative: float, after_relative: float, trim: float) -> str:
    if trim == 0.0 and TARGET_BOTTOM_MIN_DB <= before_relative <= TARGET_BOTTOM_MAX_DB:
        return "snare_bottom_already_below_top"
    if trim == 0.0:
        return "snare_bottom_not_trimmed_because_it_is_not_above_target"
    return (
        "snare_bottom trimmed below snare_top; "
        f"relative {before_relative:.2f} dB -> {after_relative:.2f} dB"
    )


def _append_snare_balance_note(top: Any, bottom: Any, report: dict[str, Any]) -> None:
    top_note = {
        "type": SNARE_TOP_BOTTOM_BALANCE_STAGE,
        "pair_role": "snare_top",
        "bottom_file": report["bottom_file"],
        "bottom_relative_after_db": report["primary_relative_after_db"],
        "reason": report["reason"],
    }
    bottom_note = {
        "type": SNARE_TOP_BOTTOM_BALANCE_STAGE,
        "pair_role": "snare_bottom",
        "top_file": report["top_file"],
        "applied_trim_db": report["applied_trim_db"],
        "bottom_relative_after_db": report["primary_relative_after_db"],
        "reason": report["reason"],
    }
    if hasattr(top, "output_balance_notes"):
        top.output_balance_notes.append(top_note)
    if hasattr(bottom, "output_balance_notes"):
        bottom.output_balance_notes.append(bottom_note)
