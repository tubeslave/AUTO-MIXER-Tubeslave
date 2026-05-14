"""Post-balance stereo width stage for the Ayaic offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import amp_to_db, db_to_amp, ensure_stereo


MUSICAL_PANORAMA_WIDTH_STAGE = "musical_panorama_width"

RoleFn = Callable[[str], str]
LufsFn = Callable[[np.ndarray, int], float]
PeakFn = Callable[[np.ndarray], float]

ROLE_WIDTH_DB = {
    "overhead": 6.0,
    "room": 4.5,
    "playback": 5.5,
    "guitar": 5.0,
    "keys": 3.5,
    "pad": 4.0,
    "backing_vocal": 3.5,
}


def apply_musical_panorama_width(
    stems: list[Any],
    *,
    report_only: bool = False,
    role_fn: RoleFn | None = None,
    lufs_fn: LufsFn | None = None,
    peak_fn: PeakFn | None = None,
    max_peak_growth_db: float = 1.5,
    max_lr_imbalance_db: float = 1.5,
) -> dict[str, Any]:
    """Increase side energy after output balance without changing mono sum.

    The panning stage chooses positions. Output balance may then pull wide
    sources down to protect the vocal. This stage restores stereo audibility by
    boosting side components only on already-panned musical bed/image sources.
    Mid is left unchanged, so mono fold-down level remains stable.
    """

    if not stems:
        return {"enabled": False, "applied": False, "stage": MUSICAL_PANORAMA_WIDTH_STAGE, "reason": "no_stems"}

    sample_rate = int(stems[0].sample_rate)
    roles = {
        int(stem.channel_id): _normalize_role(role_fn(stem.name) if role_fn else _fallback_role(stem.name))
        for stem in stems
    }
    before_audio = _sum_stems(stems)
    before_peak = _call_peak(peak_fn, before_audio)
    before_lufs = _call_lufs(lufs_fn, before_audio, sample_rate)
    before_mono_lufs = _call_lufs(lufs_fn, _mono_stereo(before_audio), sample_rate)
    before_side_mid = _side_mid_db(before_audio)
    width_targets = _width_targets(stems, roles, before_side_mid)

    if not any(abs(value) >= 0.1 for value in width_targets.values()):
        return {
            "enabled": True,
            "applied": False,
            "stage": MUSICAL_PANORAMA_WIDTH_STAGE,
            "reason": "no_wide_candidates",
            "pre_side_mid_db": round(before_side_mid, 3),
            "channel_width": [],
            "blocked_width": [],
        }

    scale, predicted, blocked = _safe_width_scale(
        stems,
        width_targets,
        before_peak=before_peak,
        max_peak_growth_db=max_peak_growth_db,
        max_lr_imbalance_db=max_lr_imbalance_db,
        peak_fn=peak_fn,
    )

    channel_reports: list[dict[str, Any]] = []
    for stem in stems:
        channel = int(stem.channel_id)
        role = roles[channel]
        requested_width = float(width_targets.get(channel, 0.0))
        final_width = requested_width * scale
        if abs(final_width) < 0.1:
            final_width = 0.0
        before_channel_peak = _call_peak(peak_fn, stem.audio)
        if not report_only and abs(final_width) >= 0.1:
            stem.audio = _apply_side_width(stem.audio, final_width)
        after_channel_peak = _call_peak(peak_fn, stem.audio)
        report = {
            "stage": MUSICAL_PANORAMA_WIDTH_STAGE,
            "channel": channel,
            "file": Path(stem.path).name,
            "role": role,
            "pan": round(float(getattr(stem, "pan", 0.0)), 3),
            "requested_side_gain_db": round(requested_width, 3),
            "final_side_gain_db": round(final_width, 3),
            "pre_peak_dbfs": round(before_channel_peak, 3),
            "post_peak_dbfs": round(after_channel_peak, 3),
            "applied": bool(not report_only and abs(final_width) >= 0.1),
            "report_only": bool(report_only),
            "reason": _width_reason(role, final_width),
        }
        _append_width_note(stem, report)
        channel_reports.append(report)

    after_audio = _sum_stems(stems)
    after_peak = _call_peak(peak_fn, after_audio)
    after_lufs = _call_lufs(lufs_fn, after_audio, sample_rate)
    after_mono_lufs = _call_lufs(lufs_fn, _mono_stereo(after_audio), sample_rate)
    after_side_mid = _side_mid_db(after_audio)

    return {
        "enabled": True,
        "applied": any(item["applied"] for item in channel_reports) and not report_only,
        "stage": MUSICAL_PANORAMA_WIDTH_STAGE,
        "method": "post_balance_mid_side_width_restore",
        "pre_mix_lufs": round(before_lufs, 3),
        "post_mix_lufs": round(after_lufs, 3),
        "pre_mix_peak_dbfs": round(before_peak, 3),
        "post_mix_peak_dbfs": round(after_peak, 3),
        "pre_mono_lufs": round(before_mono_lufs, 3),
        "post_mono_lufs": round(after_mono_lufs, 3),
        "pre_side_mid_db": round(before_side_mid, 3),
        "post_side_mid_db": round(after_side_mid, 3),
        "side_mid_delta_db": round(after_side_mid - before_side_mid, 3),
        "safety_scale": round(float(scale), 4),
        "predicted_full_width": predicted,
        "channel_width": channel_reports,
        "blocked_width": blocked,
        "critic": {
            "decision": "ok" if after_peak <= before_peak + max_peak_growth_db + 0.05 else "review",
            "mix_peak_delta_db": round(after_peak - before_peak, 3),
            "mono_foldown_delta_lufs": round(after_mono_lufs - before_mono_lufs, 3),
        },
    }


def _width_targets(stems: list[Any], roles: dict[int, str], side_mid_db: float) -> dict[int, float]:
    extra = 0.0
    if side_mid_db < -24.0:
        extra = 1.5
    elif side_mid_db < -21.0:
        extra = 0.8

    targets: dict[int, float] = {}
    for stem in stems:
        channel = int(stem.channel_id)
        role = roles[channel]
        pan = abs(float(getattr(stem, "pan", 0.0)))
        base = float(ROLE_WIDTH_DB.get(role, 0.0))
        if base <= 0.0 or pan < 0.18:
            targets[channel] = 0.0
            continue
        pan_factor = min(1.0, max(0.35, pan / 0.72))
        targets[channel] = min(7.0, (base + extra) * pan_factor)
    return targets


def _safe_width_scale(
    stems: list[Any],
    targets: dict[int, float],
    *,
    before_peak: float,
    max_peak_growth_db: float,
    max_lr_imbalance_db: float,
    peak_fn: PeakFn | None,
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    target_peak = before_peak + max_peak_growth_db

    def render(scale: float) -> tuple[np.ndarray, float, float]:
        audio = _sum_audio([
            _apply_side_width(stem.audio, float(targets.get(int(stem.channel_id), 0.0)) * scale)
            for stem in stems
        ])
        return audio, _call_peak(peak_fn, audio), abs(_lr_imbalance_db(audio))

    full_audio, full_peak, full_lr = render(1.0)
    full_side_mid = _side_mid_db(full_audio)
    predicted = {
        "peak_dbfs": round(full_peak, 3),
        "lr_imbalance_db": round(full_lr, 3),
        "side_mid_db": round(full_side_mid, 3),
        "target_peak_dbfs": round(target_peak, 3),
    }
    if full_peak <= target_peak and full_lr <= max_lr_imbalance_db:
        return 1.0, predicted, []

    low = 0.0
    high = 1.0
    best_scale = 0.0
    best_peak = before_peak
    best_lr = 0.0
    for _ in range(14):
        scale = (low + high) * 0.5
        _, peak, lr = render(scale)
        if peak <= target_peak and lr <= max_lr_imbalance_db:
            best_scale = scale
            best_peak = peak
            best_lr = lr
            low = scale
        else:
            high = scale

    blocked: list[dict[str, Any]] = []
    if best_scale < 0.999:
        blocked.append(
            {
                "role": "panorama_width_guard",
                "requested_scale": 1.0,
                "safe_scale": round(float(best_scale), 4),
                "reasons": _width_guard_reasons(full_peak, full_lr, target_peak, max_lr_imbalance_db),
                "predicted_peak_dbfs": round(full_peak, 3),
                "safe_predicted_peak_dbfs": round(best_peak, 3),
                "target_peak_dbfs": round(target_peak, 3),
                "predicted_lr_imbalance_db": round(full_lr, 3),
                "safe_lr_imbalance_db": round(best_lr, 3),
            }
        )
    return best_scale, predicted, blocked


def _width_guard_reasons(
    full_peak: float,
    full_lr: float,
    target_peak: float,
    max_lr: float,
) -> list[str]:
    reasons = []
    if full_peak > target_peak:
        reasons.append("mix_peak_growth_limited")
    if full_lr > max_lr:
        reasons.append("lr_imbalance_limited")
    return reasons or ["ok"]


def _apply_side_width(audio: np.ndarray, side_gain_db: float) -> np.ndarray:
    arr = ensure_stereo(audio).astype(np.float32, copy=False)
    if abs(side_gain_db) < 0.01:
        return arr.copy()
    mid = (arr[:, 0] + arr[:, 1]) * 0.5
    side = (arr[:, 0] - arr[:, 1]) * 0.5 * db_to_amp(side_gain_db)
    return np.column_stack([mid + side, mid - side]).astype(np.float32)


def _append_width_note(stem: Any, report: dict[str, Any]) -> None:
    note = {
        "type": MUSICAL_PANORAMA_WIDTH_STAGE,
        "role": report["role"],
        "final_side_gain_db": report["final_side_gain_db"],
        "reason": report["reason"],
    }
    if hasattr(stem, "pan_notes"):
        stem.pan_notes.append(note)


def _width_reason(role: str, width_db: float) -> str:
    if abs(width_db) < 0.1:
        return f"{role} width held by panorama safety"
    return f"{role} side restored after output balance"


def _side_mid_db(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    mid = (arr[:, 0] + arr[:, 1]) * 0.5
    side = (arr[:, 0] - arr[:, 1]) * 0.5
    return amp_to_db(_rms(side) / max(_rms(mid), 1e-12))


def _lr_imbalance_db(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return 0.0
    return amp_to_db(_rms(arr[:, 0]) / max(_rms(arr[:, 1]), 1e-12))


def _mono_stereo(audio: np.ndarray) -> np.ndarray:
    arr = ensure_stereo(audio)
    mono = np.mean(arr, axis=1)
    return np.column_stack([mono, mono]).astype(np.float32)


def _sum_stems(stems: list[Any]) -> np.ndarray:
    return _sum_audio([stem.audio for stem in stems])


def _sum_audio(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.zeros((0, 2), dtype=np.float32)
    max_len = max(ensure_stereo(item).shape[0] for item in items)
    out = np.zeros((max_len, 2), dtype=np.float32)
    for item in items:
        audio = ensure_stereo(item)
        if audio.shape[0] < max_len:
            audio = np.pad(audio, ((0, max_len - audio.shape[0]), (0, 0)))
        out += audio[:max_len]
    return out


def _call_lufs(fn: LufsFn | None, audio: np.ndarray, sample_rate: int) -> float:
    if fn is None:
        return amp_to_db(_rms(audio))
    value = float(fn(audio, sample_rate))
    return value if np.isfinite(value) else -120.0


def _call_peak(fn: PeakFn | None, audio: np.ndarray) -> float:
    if fn is None:
        return amp_to_db(float(np.max(np.abs(audio))) if audio.size else 0.0)
    value = float(fn(audio))
    return value if np.isfinite(value) else -120.0


def _rms(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr) + 1e-12))


def _normalize_role(role: str) -> str:
    key = str(role or "unknown").strip().lower()
    if key in {"lead_vocal", "vocal", "vox"}:
        return "lead_vocal"
    if key in {"back_vocal", "backing_vocal", "backs", "bgv"}:
        return "backing_vocal"
    if key in {"snare", "snare_top", "snare_bottom"}:
        return "snare"
    if key in {"tom", "rack_tom", "floor_tom"}:
        return "tom"
    if key in {"bass", "bass_guitar"}:
        return "bass"
    if key in {"guitar", "rhythm_guitar", "lead_guitar", "electric_guitar", "acoustic_guitar"}:
        return "guitar"
    if key in {"accordion", "keys", "piano", "synth"}:
        return "keys"
    if key in {"overhead", "room", "kick", "pad", "playback"}:
        return key
    return "unknown"


def _fallback_role(name: str) -> str:
    key = Path(str(name)).stem.lower()
    if "oh" in key or "overhead" in key:
        return "overhead"
    if "room" in key:
        return "room"
    if "playback" in key:
        return "playback"
    if "back" in key or "backs" in key:
        return "backing_vocal"
    if "guitar" in key or "gtr" in key:
        return "guitar"
    if "accordion" in key or "keys" in key or "piano" in key:
        return "keys"
    if "pad" in key:
        return "pad"
    return "unknown"
