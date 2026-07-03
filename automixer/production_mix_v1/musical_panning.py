"""Musical panning solver for the Ayaic offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import amp_to_db, db_to_amp, ensure_stereo


MUSICAL_PANNING_STAGE = "musical_panning"
DEFAULT_MUSICAL_PANNING_STYLE = "live_pop_rock"

RoleFn = Callable[[str], str]
AudioFn = Callable[[Any], np.ndarray]
LufsFn = Callable[[np.ndarray, int], float]
PeakFn = Callable[[np.ndarray], float]
BandEnergyFn = Callable[[np.ndarray, int], dict[str, float]]


@dataclass(frozen=True)
class PanningMetrics:
    full_lufs: float
    peak_dbfs: float
    activity_ratio: float
    low_energy_db: float
    low_mid_db: float
    presence_db: float
    stereo_correlation: float
    dual_mono: bool


ROLE_PAN_RULES: dict[str, dict[str, float | str | bool]] = {
    "lead_vocal": {"base_pan": 0.0, "max_abs_pan": 0.08, "priority": 0, "zone": "center", "center_locked": True},
    "backing_vocal": {"base_pan": 0.32, "max_abs_pan": 0.55, "priority": 3, "zone": "support"},
    "kick": {"base_pan": 0.0, "max_abs_pan": 0.04, "priority": 0, "zone": "center", "center_locked": True},
    "bass": {"base_pan": 0.0, "max_abs_pan": 0.04, "priority": 0, "zone": "center", "center_locked": True},
    "snare": {"base_pan": 0.0, "max_abs_pan": 0.04, "priority": 1, "zone": "center", "center_locked": True},
    "tom": {"base_pan": 0.30, "max_abs_pan": 0.62, "priority": 4, "zone": "drums"},
    "overhead": {"base_pan": 0.72, "max_abs_pan": 0.82, "priority": 5, "zone": "image_air"},
    "room": {"base_pan": 0.58, "max_abs_pan": 0.72, "priority": 7, "zone": "image_air"},
    "guitar": {"base_pan": 0.46, "max_abs_pan": 0.70, "priority": 5, "zone": "music_bed"},
    "keys": {"base_pan": 0.24, "max_abs_pan": 0.55, "priority": 5, "zone": "music_bed"},
    "pad": {"base_pan": 0.34, "max_abs_pan": 0.70, "priority": 7, "zone": "music_bed"},
    "playback": {"base_pan": 0.68, "max_abs_pan": 0.82, "priority": 6, "zone": "music_bed"},
    "unknown": {"base_pan": 0.0, "max_abs_pan": 0.18, "priority": 9, "zone": "unknown"},
}


STYLE_MODIFIERS: dict[str, dict[str, float]] = {
    "live_pop_rock": {"width": 1.0, "bed_spread": 1.0, "max_lr_imbalance_db": 1.5, "max_mono_loss_db": 2.4},
    "rock": {"width": 0.9, "bed_spread": 0.85, "max_lr_imbalance_db": 1.7, "max_mono_loss_db": 1.6},
    "wide": {"width": 1.12, "bed_spread": 1.1, "max_lr_imbalance_db": 1.7, "max_mono_loss_db": 1.4},
    "narrow": {"width": 0.72, "bed_spread": 0.7, "max_lr_imbalance_db": 1.2, "max_mono_loss_db": 0.9},
    "transparent": {"width": 0.82, "bed_spread": 0.78, "max_lr_imbalance_db": 1.2, "max_mono_loss_db": 0.9},
}


def apply_musical_panning(
    stems: list[Any],
    *,
    style: str = DEFAULT_MUSICAL_PANNING_STYLE,
    report_only: bool = False,
    role_fn: RoleFn | None = None,
    primary_audio_fn: AudioFn | None = None,
    lufs_fn: LufsFn | None = None,
    peak_fn: PeakFn | None = None,
    band_energy_fn: BandEnergyFn | None = None,
) -> dict[str, Any]:
    """Apply a role-aware musical pan scene before final output balancing."""
    style_key = _style_key(style)
    if not stems:
        return {"enabled": False, "applied": False, "stage": MUSICAL_PANNING_STAGE, "reason": "no_stems"}

    sample_rate = int(stems[0].sample_rate)
    roles: dict[int, str] = {}
    metrics: dict[int, PanningMetrics] = {}
    for stem in stems:
        channel = int(stem.channel_id)
        role = _normalize_role(role_fn(stem.name) if role_fn else _fallback_role(stem.name))
        roles[channel] = role
        metrics[channel] = _measure_metrics(
            stem,
            primary_audio_fn=primary_audio_fn,
            lufs_fn=lufs_fn,
            peak_fn=peak_fn,
            band_energy_fn=band_energy_fn,
        )

    pairs = _detect_stereo_pairs(stems, roles)
    raw_pans = _raw_pan_targets(stems, roles, metrics, pairs, style=style_key)
    safe_pans, blocked = _apply_pan_safety(stems, roles, metrics, raw_pans, pairs)

    before_mix = _sum_stems(stems)
    before_mix_lufs = _call_lufs(lufs_fn, before_mix, sample_rate)
    before_mix_peak = _call_peak(peak_fn, before_mix)
    before_lr_imbalance = _lr_imbalance_db(before_mix)
    before_mono_lufs = _call_lufs(lufs_fn, _mono_stereo(before_mix), sample_rate)

    safe_pans, global_blocks = _limit_scene_risks(
        stems,
        safe_pans,
        before_mix=before_mix,
        before_peak=before_mix_peak,
        before_mono_lufs=before_mono_lufs,
        style=style_key,
        lufs_fn=lufs_fn,
        peak_fn=peak_fn,
    )
    blocked.extend(global_blocks)

    channel_reports: list[dict[str, Any]] = []
    for stem in stems:
        channel = int(stem.channel_id)
        role = roles[channel]
        metric = metrics[channel]
        before_pan = float(getattr(stem, "pan", 0.0))
        final_pan = float(safe_pans.get(channel, before_pan))
        if not report_only and abs(final_pan) > 1e-6:
            stem.audio = _apply_pan_preserve_width(stem.audio, final_pan, dual_mono=metric.dual_mono)
        if not report_only and abs(final_pan - before_pan) > 1e-6:
            stem.pan = final_pan
        report = {
            "stage": MUSICAL_PANNING_STAGE,
            "channel": channel,
            "file": Path(stem.path).name,
            "role": role,
            "zone": str(_role_rule(role)["zone"]),
            "stereo_pair": _pair_name_for_channel(channel, pairs),
            "before_pan": round(before_pan, 3),
            "requested_pan": round(raw_pans.get(channel, before_pan), 3),
            "final_pan": round(final_pan, 3),
            "pan_100": round(final_pan * 100.0, 1),
            "activity_ratio": round(metric.activity_ratio, 4),
            "low_energy_db": round(metric.low_energy_db, 3),
            "presence_db": round(metric.presence_db, 3),
            "stereo_correlation": round(metric.stereo_correlation, 3),
            "dual_mono": bool(metric.dual_mono),
            "applied": bool(not report_only and abs(final_pan) >= 0.01),
            "report_only": bool(report_only),
            "reason": _pan_reason(role, before_pan, final_pan, pairs.get(channel)),
        }
        _append_pan_note(stem, report)
        channel_reports.append(report)

    after_mix = _sum_stems(stems)
    after_mix_lufs = _call_lufs(lufs_fn, after_mix, sample_rate)
    after_mix_peak = _call_peak(peak_fn, after_mix)
    after_lr_imbalance = _lr_imbalance_db(after_mix)
    after_mono_lufs = _call_lufs(lufs_fn, _mono_stereo(after_mix), sample_rate)
    critic = _critic_report(
        before_peak=before_mix_peak,
        after_peak=after_mix_peak,
        before_lr_imbalance=before_lr_imbalance,
        after_lr_imbalance=after_lr_imbalance,
        before_mono_lufs=before_mono_lufs,
        after_mono_lufs=after_mono_lufs,
        channel_reports=channel_reports,
        style=style_key,
    )

    return {
        "enabled": True,
        "applied": any(item["applied"] for item in channel_reports) and not report_only,
        "stage": MUSICAL_PANNING_STAGE,
        "method": "hybrid_role_scene_spatial_unmasking_solver",
        "style": style_key,
        "pre_mix_lufs": round(before_mix_lufs, 3),
        "post_mix_lufs": round(after_mix_lufs, 3),
        "pre_mix_peak_dbfs": round(before_mix_peak, 3),
        "post_mix_peak_dbfs": round(after_mix_peak, 3),
        "pre_lr_imbalance_db": round(before_lr_imbalance, 3),
        "post_lr_imbalance_db": round(after_lr_imbalance, 3),
        "pre_mono_lufs": round(before_mono_lufs, 3),
        "post_mono_lufs": round(after_mono_lufs, 3),
        "channel_pans": channel_reports,
        "pair_pans": _pair_report(stems, pairs, safe_pans),
        "blocked_pans": blocked,
        "critic": critic,
        "source": {
            "name": "Musical Panning",
            "source": "local role/template scene plus spatial-unmasking and mono/LR critic",
            "references": [
                {
                    "name": "A real-time semiautonomous audio panning system for music mixing",
                    "url": "https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/436895",
                    "use": "autonomous and semiautonomous panning criteria",
                },
                {
                    "name": "An autonomous system for multi-track stereo pan positioning",
                    "url": "https://www.eecs.qmul.ac.uk/~josh/documents/2012/MansbridgeFinnReiss-AES133-Autonomoussystemformultitrackstereopositioning.pdf",
                    "use": "source, spatial, and spectral balance constraints",
                },
                {
                    "name": "An automatic mixing system for multitrack spatialization for stereo based on unmasking and best panning practices",
                    "url": "https://eecs.qmul.ac.uk/~josh/documents/2019/20311.pdf",
                    "use": "unmasking-driven stereo spatialization",
                },
            ],
            "notes": [
                "This stage is offline-only in the Ayaic render path.",
                "Universal pan tables are engineering heuristics; no strong evidence supports one fixed layout for every song.",
            ],
        },
    }


def _measure_metrics(
    stem: Any,
    *,
    primary_audio_fn: AudioFn | None,
    lufs_fn: LufsFn | None,
    peak_fn: PeakFn | None,
    band_energy_fn: BandEnergyFn | None,
) -> PanningMetrics:
    audio = ensure_stereo(getattr(stem, "audio"))
    primary = ensure_stereo(primary_audio_fn(stem)) if primary_audio_fn else audio
    sample_rate = int(stem.sample_rate)
    full_lufs = _call_lufs(lufs_fn, audio, sample_rate)
    peak = _call_peak(peak_fn, audio)
    band = _call_band_energy(band_energy_fn, primary, sample_rate)
    mono = _mono(primary)
    activity = _activity_ratio(mono)
    corr = _stereo_corr(audio)
    dual_mono = _is_dual_mono(audio, corr)
    return PanningMetrics(
        full_lufs=full_lufs,
        peak_dbfs=peak,
        activity_ratio=activity,
        low_energy_db=_combined_band_db(band, ("low", "sub", "bass")),
        low_mid_db=float(band.get("low_mid", -100.0)),
        presence_db=float(band.get("presence", band.get("high_mid", -100.0))),
        stereo_correlation=corr,
        dual_mono=dual_mono,
    )


def _raw_pan_targets(
    stems: list[Any],
    roles: dict[int, str],
    metrics: dict[int, PanningMetrics],
    pairs: dict[int, dict[str, Any]],
    *,
    style: str,
) -> dict[int, float]:
    modifier = STYLE_MODIFIERS[_style_key(style)]
    width = float(modifier["width"])
    bed_spread = float(modifier["bed_spread"])
    out: dict[int, float] = {}
    side_counts = {"left": 0, "right": 0}
    for stem in sorted(stems, key=lambda item: int(item.channel_id)):
        channel = int(stem.channel_id)
        role = roles[channel]
        rule = _role_rule(role)
        pair = pairs.get(channel)
        if bool(rule.get("center_locked", False)):
            out[channel] = 0.0
            continue
        if pair:
            sign = -1.0 if pair["side"] == "left" else 1.0
            pair_role = str(pair["role"])
            pair_rule = _role_rule(pair_role)
            out[channel] = sign * float(pair_rule["base_pan"]) * width
            continue
        sign = _name_side_hint(stem.name)
        if sign == 0.0:
            sign = _next_scene_side(side_counts)
        zone = str(rule["zone"])
        spread = bed_spread if zone in {"music_bed", "support"} else 1.0
        target = sign * float(rule["base_pan"]) * width * spread
        if role in {"tom", "overhead", "room"}:
            target = _drum_name_target(stem.name, fallback=target, width=width)
        if role in {"guitar", "keys", "pad", "playback"} and _presence_masks_lead(metrics[channel], metrics, roles):
            target = _push_away_from_center(target, amount=0.10 * bed_spread)
        out[channel] = target
    return out


def _detect_stereo_pairs(stems: list[Any], roles: dict[int, str]) -> dict[int, dict[str, Any]]:
    by_base: dict[str, dict[str, Any]] = {}
    for stem in stems:
        side = _name_side(stem.name)
        if side is None:
            continue
        base = _pair_base(stem.name)
        if not base:
            continue
        by_base.setdefault(base, {})[side] = stem
    pairs: dict[int, dict[str, Any]] = {}
    for base, sides in by_base.items():
        left = sides.get("left")
        right = sides.get("right")
        if left is None or right is None:
            continue
        left_ch = int(left.channel_id)
        right_ch = int(right.channel_id)
        role = roles.get(left_ch, roles.get(right_ch, "unknown"))
        if role == "unknown":
            role = roles.get(right_ch, "unknown")
        pair_id = f"{base}_lr"
        pairs[left_ch] = {"pair_id": pair_id, "base": base, "side": "left", "mate_channel": right_ch, "role": role}
        pairs[right_ch] = {"pair_id": pair_id, "base": base, "side": "right", "mate_channel": left_ch, "role": role}
    return pairs


def _apply_pan_safety(
    stems: list[Any],
    roles: dict[int, str],
    metrics: dict[int, PanningMetrics],
    targets: dict[int, float],
    pairs: dict[int, dict[str, Any]],
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    safe: dict[int, float] = {}
    blocked: list[dict[str, Any]] = []
    for stem in stems:
        channel = int(stem.channel_id)
        role = roles[channel]
        rule = _role_rule(role)
        requested = float(targets.get(channel, getattr(stem, "pan", 0.0)))
        max_abs = float(rule["max_abs_pan"])
        limited = float(np.clip(requested, -max_abs, max_abs))
        reasons: list[str] = []
        if abs(limited - requested) > 1e-6:
            reasons.append("role_pan_limit")
        if role in {"bass", "kick"} and abs(limited) > 0.04:
            limited = 0.0
            reasons.append("low_end_center_lock")
        if _low_end_dominant(metrics[channel]):
            low_end_limit = _low_end_pan_limit(role, pairs.get(channel))
            if abs(limited) > low_end_limit:
                limited = float(np.clip(limited, -low_end_limit, low_end_limit))
                reasons.append("low_end_energy_center_guard")
        if role == "unknown" and abs(limited) > 0.18:
            limited = float(np.clip(limited, -0.18, 0.18))
            reasons.append("unknown_channel_pan_limit")
        if abs(limited) < 0.01:
            limited = 0.0
        safe[channel] = limited
        if reasons:
            blocked.append(
                {
                    "channel": channel,
                    "file": Path(stem.path).name,
                    "role": role,
                    "requested_pan": round(requested, 3),
                    "safe_pan": round(limited, 3),
                    "reasons": reasons,
                }
            )
    _enforce_pair_symmetry(stems, pairs, safe, blocked)
    return safe, blocked


def _enforce_pair_symmetry(
    stems: list[Any],
    pairs: dict[int, dict[str, Any]],
    safe: dict[int, float],
    blocked: list[dict[str, Any]],
) -> None:
    seen: set[str] = set()
    by_channel = {int(stem.channel_id): stem for stem in stems}
    for channel, pair in sorted(pairs.items()):
        pair_id = str(pair["pair_id"])
        if pair_id in seen:
            continue
        mate = int(pair["mate_channel"])
        if channel not in by_channel or mate not in by_channel:
            continue
        left = channel if pair["side"] == "left" else mate
        right = mate if pair["side"] == "left" else channel
        width = min(abs(float(safe.get(left, 0.0))), abs(float(safe.get(right, 0.0))))
        new_left = -width if width >= 0.01 else 0.0
        new_right = width if width >= 0.01 else 0.0
        old_left = float(safe.get(left, 0.0))
        old_right = float(safe.get(right, 0.0))
        safe[left] = new_left
        safe[right] = new_right
        seen.add(pair_id)
        if abs(old_left - new_left) > 1e-6 or abs(old_right - new_right) > 1e-6:
            blocked.append(
                {
                    "pair_id": pair_id,
                    "role": "stereo_pair_symmetry_guard",
                    "left_channel": left,
                    "right_channel": right,
                    "left_file": Path(by_channel[left].path).name,
                    "right_file": Path(by_channel[right].path).name,
                    "requested_left_pan": round(old_left, 3),
                    "requested_right_pan": round(old_right, 3),
                    "safe_left_pan": round(new_left, 3),
                    "safe_right_pan": round(new_right, 3),
                    "reasons": ["stereo_pair_symmetry_guard"],
                }
            )


def _limit_scene_risks(
    stems: list[Any],
    targets: dict[int, float],
    *,
    before_mix: np.ndarray,
    before_peak: float,
    before_mono_lufs: float,
    style: str,
    lufs_fn: LufsFn | None,
    peak_fn: PeakFn | None,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    modifier = STYLE_MODIFIERS[_style_key(style)]
    max_lr = float(modifier["max_lr_imbalance_db"])
    max_mono_loss = float(modifier["max_mono_loss_db"])
    target_peak = before_peak + 0.25
    sample_rate = int(stems[0].sample_rate) if stems else 48_000
    # In the offline render path, existing stem.pan can be metadata from name
    # detection. The audio itself has not yet been panned, so risk prediction
    # interpolates from the rendered center signal.
    current = {int(stem.channel_id): 0.0 for stem in stems}

    def candidate(scale: float) -> tuple[dict[int, float], np.ndarray, float, float, float]:
        pans = {
            channel: current.get(channel, 0.0) + (float(targets.get(channel, current.get(channel, 0.0))) - current.get(channel, 0.0)) * scale
            for channel in targets
        }
        rendered = []
        for stem in stems:
            channel = int(stem.channel_id)
            pan = pans.get(channel, current.get(channel, 0.0))
            if abs(pan - current.get(channel, 0.0)) <= 1e-6:
                rendered.append(ensure_stereo(stem.audio))
            else:
                rendered.append(_apply_pan_preserve_width(stem.audio, pan, dual_mono=_is_dual_mono(stem.audio)))
        mix = _sum_audio(rendered)
        peak = _call_peak(peak_fn, mix)
        lr = abs(_lr_imbalance_db(mix))
        mono_lufs = _call_lufs(lufs_fn, _mono_stereo(mix), sample_rate)
        return pans, mix, peak, lr, mono_lufs

    _, _, full_peak, full_lr, full_mono_lufs = candidate(1.0)
    full_mono_loss = before_mono_lufs - full_mono_lufs
    if full_peak <= target_peak and full_lr <= max_lr and full_mono_loss <= max_mono_loss:
        return targets, []

    low = 0.0
    high = 1.0
    best_scale = 0.0
    best: tuple[dict[int, float], float, float, float] | None = None
    for _ in range(14):
        scale = (low + high) * 0.5
        pans, _, peak, lr, mono_lufs = candidate(scale)
        mono_loss = before_mono_lufs - mono_lufs
        if peak <= target_peak and lr <= max_lr and mono_loss <= max_mono_loss:
            best_scale = scale
            best = (pans, peak, lr, mono_loss)
            low = scale
        else:
            high = scale

    if best is None:
        scaled = current
        safe_peak = before_peak
        safe_lr = abs(_lr_imbalance_db(before_mix))
        safe_mono_loss = 0.0
    else:
        scaled, safe_peak, safe_lr, safe_mono_loss = best

    blocked: list[dict[str, Any]] = []
    reasons = []
    if full_peak > target_peak:
        reasons.append("mix_peak_growth_limited")
    if full_lr > max_lr:
        reasons.append("lr_imbalance_limited")
    if full_mono_loss > max_mono_loss:
        reasons.append("mono_foldown_loss_limited")
    for stem in stems:
        channel = int(stem.channel_id)
        requested = float(targets.get(channel, current.get(channel, 0.0)))
        safe = float(scaled.get(channel, current.get(channel, 0.0)))
        if abs(requested - safe) > 1e-3:
            blocked.append(
                {
                    "channel": channel,
                    "file": Path(stem.path).name,
                    "role": "scene_risk_guard",
                    "requested_pan": round(requested, 3),
                    "safe_pan": round(safe, 3),
                    "scale": round(float(best_scale), 4),
                    "reasons": reasons,
                    "predicted_peak_dbfs": round(full_peak, 3),
                    "safe_predicted_peak_dbfs": round(float(safe_peak), 3),
                    "target_peak_dbfs": round(target_peak, 3),
                    "predicted_lr_imbalance_db": round(float(full_lr), 3),
                    "safe_lr_imbalance_db": round(float(safe_lr), 3),
                    "predicted_mono_loss_db": round(float(full_mono_loss), 3),
                    "safe_mono_loss_db": round(float(safe_mono_loss), 3),
                }
            )
    return scaled, blocked


def _critic_report(
    *,
    before_peak: float,
    after_peak: float,
    before_lr_imbalance: float,
    after_lr_imbalance: float,
    before_mono_lufs: float,
    after_mono_lufs: float,
    channel_reports: list[dict[str, Any]],
    style: str,
) -> dict[str, Any]:
    modifier = STYLE_MODIFIERS[_style_key(style)]
    notes: list[dict[str, Any]] = []
    if after_peak > before_peak + 0.25:
        notes.append({"severity": "medium", "reason": "mix_peak_increased", "delta_db": round(after_peak - before_peak, 3)})
    if abs(after_lr_imbalance) > float(modifier["max_lr_imbalance_db"]) + 0.05:
        notes.append({"severity": "medium", "reason": "left_right_imbalance", "imbalance_db": round(after_lr_imbalance, 3)})
    mono_loss = before_mono_lufs - after_mono_lufs
    if mono_loss > float(modifier["max_mono_loss_db"]) + 0.05:
        notes.append({"severity": "medium", "reason": "mono_foldown_loss", "loss_db": round(mono_loss, 3)})
    center_violations = [
        {"channel": item["channel"], "file": item["file"], "pan": item["final_pan"], "role": item["role"]}
        for item in channel_reports
        if item["role"] in {"lead_vocal", "kick", "bass", "snare"} and abs(float(item["final_pan"])) > 0.08
    ]
    if center_violations:
        notes.append({"severity": "high", "reason": "center_anchor_moved", "channels": center_violations})
    return {
        "decision": "ok" if not any(note["severity"] == "high" for note in notes) else "review",
        "notes": notes,
        "mix_peak_delta_db": round(after_peak - before_peak, 3),
        "lr_imbalance_delta_db": round(after_lr_imbalance - before_lr_imbalance, 3),
        "mono_foldown_delta_lufs": round(after_mono_lufs - before_mono_lufs, 3),
    }


def _apply_pan_preserve_width(audio: np.ndarray, pan: float, *, dual_mono: bool | None = None) -> np.ndarray:
    arr = ensure_stereo(audio).astype(np.float32, copy=False)
    pan = float(np.clip(pan, -1.0, 1.0))
    if dual_mono is None:
        dual_mono = _is_dual_mono(arr)
    if dual_mono:
        mono = _mono(arr)
        left_scale, right_scale = _balance_scales(pan)
        return np.column_stack([mono * left_scale, mono * right_scale]).astype(np.float32)
    left_scale, right_scale = _balance_scales(pan)
    return np.column_stack([arr[:, 0] * left_scale, arr[:, 1] * right_scale]).astype(np.float32)


def _equal_power_gains(pan: float) -> tuple[float, float]:
    theta = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    return float(np.cos(theta)), float(np.sin(theta))


def _balance_scales(pan: float) -> tuple[float, float]:
    pan = float(np.clip(pan, -1.0, 1.0))
    if pan < 0.0:
        return 1.0, float(np.cos(abs(pan) * np.pi / 2.0))
    if pan > 0.0:
        return float(np.cos(pan * np.pi / 2.0)), 1.0
    return 1.0, 1.0


def _pair_report(stems: list[Any], pairs: dict[int, dict[str, Any]], pans: dict[int, float]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    by_channel = {int(stem.channel_id): stem for stem in stems}
    for channel, pair in sorted(pairs.items()):
        pair_id = str(pair["pair_id"])
        if pair_id in seen:
            continue
        mate = int(pair["mate_channel"])
        if mate not in by_channel or channel not in by_channel:
            continue
        left_ch = channel if pair["side"] == "left" else mate
        right_ch = mate if pair["side"] == "left" else channel
        left = by_channel[left_ch]
        right = by_channel[right_ch]
        seen.add(pair_id)
        out.append(
            {
                "pair_id": pair_id,
                "base": pair["base"],
                "role": pair["role"],
                "left_channel": left_ch,
                "right_channel": right_ch,
                "left_file": Path(left.path).name,
                "right_file": Path(right.path).name,
                "left_pan": round(float(pans.get(left_ch, 0.0)), 3),
                "right_pan": round(float(pans.get(right_ch, 0.0)), 3),
                "width": round(float(pans.get(right_ch, 0.0) - pans.get(left_ch, 0.0)), 3),
            }
        )
    return out


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
    key = _name_key(name)
    if "vocal" in key or "vox" in key:
        return "backing_vocal" if any(token in key for token in ("back", "bv", "bgv")) else "lead_vocal"
    if "kick" in key or "bd" in key:
        return "kick"
    if "snare" in key or "sd" in key:
        return "snare"
    if "tom" in key:
        return "tom"
    if "oh" in key or "overhead" in key:
        return "overhead"
    if "room" in key:
        return "room"
    if "bass" in key:
        return "bass"
    if "gtr" in key or "guitar" in key:
        return "guitar"
    if "key" in key or "piano" in key or "synth" in key or "accordion" in key:
        return "keys"
    if "playback" in key:
        return "playback"
    return "unknown"


def _role_rule(role: str) -> dict[str, float | str | bool]:
    return ROLE_PAN_RULES.get(role, ROLE_PAN_RULES["unknown"])


def _style_key(style: str) -> str:
    key = str(style or DEFAULT_MUSICAL_PANNING_STYLE).strip().lower()
    return key if key in STYLE_MODIFIERS else DEFAULT_MUSICAL_PANNING_STYLE


def _name_key(name: str) -> str:
    return " ".join(Path(str(name)).stem.lower().replace("_", " ").replace("-", " ").split())


def _name_side(name: str) -> str | None:
    key = _name_key(name)
    tokens = key.split()
    if tokens and tokens[-1] in {"l", "left"}:
        return "left"
    if tokens and tokens[-1] in {"r", "right"}:
        return "right"
    return None


def _name_side_hint(name: str) -> float:
    side = _name_side(name)
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    key = _name_key(name)
    if "floor tom" in key or "f tom" in key:
        return 1.0
    if "rack tom" in key:
        return -1.0
    return 0.0


def _pair_base(name: str) -> str:
    tokens = _name_key(name).split()
    if tokens and tokens[-1] in {"l", "r", "left", "right"}:
        return " ".join(tokens[:-1])
    return ""


def _next_scene_side(side_counts: dict[str, int]) -> float:
    if side_counts["left"] <= side_counts["right"]:
        side_counts["left"] += 1
        return -1.0
    side_counts["right"] += 1
    return 1.0


def _drum_name_target(name: str, *, fallback: float, width: float) -> float:
    key = _name_key(name)
    if "oh" in key or "overhead" in key:
        hint = _name_side_hint(name)
        return hint * 0.72 * width if hint else fallback
    if "floor tom" in key or "f tom" in key:
        return 0.38 * width
    if "rack tom" in key:
        return -0.30 * width
    if "tom" in key:
        return fallback
    return fallback


def _presence_masks_lead(metric: PanningMetrics, metrics: dict[int, PanningMetrics], roles: dict[int, str]) -> bool:
    lead_presence = [
        metrics[channel].presence_db
        for channel, role in roles.items()
        if role == "lead_vocal" and metrics[channel].activity_ratio > 0.01
    ]
    if not lead_presence or metric.activity_ratio <= 0.01:
        return False
    return metric.presence_db > max(lead_presence) - 8.0


def _push_away_from_center(pan: float, *, amount: float) -> float:
    if pan < 0.0:
        return pan - amount
    if pan > 0.0:
        return pan + amount
    return amount


def _low_end_dominant(metric: PanningMetrics) -> bool:
    return metric.low_energy_db > metric.presence_db + 7.0 and metric.low_energy_db > metric.low_mid_db + 3.0


def _low_end_pan_limit(role: str, pair: dict[str, Any] | None) -> float:
    if not pair:
        return 0.18
    if role == "playback":
        return 0.46
    if role == "guitar":
        return 0.36
    if role in {"keys", "pad"}:
        return 0.30
    return 0.18


def _activity_ratio(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    frame = max(512, min(4800, arr.size // 20 or 512))
    values = []
    for start in range(0, max(1, arr.size - frame + 1), frame):
        block = arr[start:start + frame]
        values.append(float(np.sqrt(np.mean(block * block) + 1e-12)))
    rms = np.asarray(values, dtype=np.float64)
    if rms.size == 0:
        return 0.0
    threshold = max(float(np.percentile(rms, 20)) * 2.0, float(np.max(rms)) * 0.08, 1e-5)
    return float(np.mean(rms > threshold))


def _call_band_energy(fn: BandEnergyFn | None, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if fn:
        try:
            result = fn(audio, sample_rate)
            if isinstance(result, dict):
                return {str(key): float(value) for key, value in result.items()}
        except Exception:
            pass
    return _fallback_band_energy(audio, sample_rate)


def _fallback_band_energy(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    mono = _mono(audio).astype(np.float64, copy=False)
    if mono.size == 0:
        return {"low": -100.0, "low_mid": -100.0, "presence": -100.0}
    window = np.hanning(mono.size)
    spectrum = np.fft.rfft(mono * window)
    freqs = np.fft.rfftfreq(mono.size, 1.0 / float(sample_rate))
    power = np.abs(spectrum) ** 2

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return -100.0
        return amp_to_db(float(np.sqrt(np.mean(power[mask]) + 1e-20)))

    return {"low": band(40.0, 160.0), "low_mid": band(180.0, 600.0), "presence": band(1500.0, 4500.0)}


def _combined_band_db(band_energy: dict[str, float], names: tuple[str, ...]) -> float:
    values = [float(band_energy.get(name, -100.0)) for name in names if name in band_energy]
    if not values:
        return -100.0
    powers = [db_to_amp(value) ** 2 for value in values]
    return amp_to_db(float(np.sqrt(sum(powers) / max(1, len(powers)))))


def _stereo_corr(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio).astype(np.float64, copy=False)
    if arr.shape[0] < 2:
        return 1.0
    left = arr[:, 0]
    right = arr[:, 1]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 1.0
    return float(np.clip(np.corrcoef(left, right)[0, 1], -1.0, 1.0))


def _is_dual_mono(audio: np.ndarray, corr: float | None = None) -> bool:
    arr = ensure_stereo(audio).astype(np.float64, copy=False)
    if arr.shape[1] < 2:
        return True
    if corr is None:
        corr = _stereo_corr(arr)
    left = float(np.sqrt(np.mean(arr[:, 0] ** 2) + 1e-12))
    right = float(np.sqrt(np.mean(arr[:, 1] ** 2) + 1e-12))
    diff_db = abs(amp_to_db(left) - amp_to_db(right))
    side = arr[:, 0] - arr[:, 1]
    side_ratio = float(np.sqrt(np.mean(side ** 2) + 1e-12) / max(left + right, 1e-12))
    return bool(corr > 0.995 and diff_db < 0.15 and side_ratio < 0.02)


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


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(ensure_stereo(audio), axis=1).astype(np.float32)


def _mono_stereo(audio: np.ndarray) -> np.ndarray:
    mono = _mono(audio)
    return np.column_stack([mono, mono]).astype(np.float32)


def _lr_imbalance_db(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio).astype(np.float64, copy=False)
    if arr.size == 0:
        return 0.0
    left = float(np.sqrt(np.mean(arr[:, 0] ** 2) + 1e-12))
    right = float(np.sqrt(np.mean(arr[:, 1] ** 2) + 1e-12))
    return amp_to_db(right) - amp_to_db(left)


def _call_lufs(fn: LufsFn | None, audio: np.ndarray, sample_rate: int) -> float:
    if fn:
        try:
            value = float(fn(audio, sample_rate))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    return _rms_db(audio) - 0.691


def _call_peak(fn: PeakFn | None, audio: np.ndarray) -> float:
    if fn:
        try:
            value = float(fn(audio))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.max(np.abs(arr))))


def _rms_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.sqrt(np.mean(arr * arr) + 1e-12)))


def _pan_reason(role: str, before_pan: float, final_pan: float, pair: dict[str, Any] | None) -> str:
    if abs(final_pan - before_pan) < 0.01:
        return f"{role} held in current panorama"
    if pair:
        return f"{role} placed as {pair['side']} side of stereo pair"
    if role in {"lead_vocal", "kick", "bass", "snare"}:
        return f"{role} kept near center as mix anchor"
    if final_pan < 0.0:
        return f"{role} moved left for spatial separation"
    return f"{role} moved right for spatial separation"


def _pair_name_for_channel(channel: int, pairs: dict[int, dict[str, Any]]) -> str | None:
    pair = pairs.get(channel)
    return str(pair["pair_id"]) if pair else None


def _append_pan_note(stem: Any, report: dict[str, Any]) -> None:
    note = {
        "stage": MUSICAL_PANNING_STAGE,
        "role": report["role"],
        "before_pan": report["before_pan"],
        "final_pan": report["final_pan"],
        "pan_100": report["pan_100"],
        "reason": report["reason"],
    }
    if hasattr(stem, "pan_notes"):
        stem.pan_notes.append(note)
    elif hasattr(stem, "notes"):
        stem.notes.append(note)
