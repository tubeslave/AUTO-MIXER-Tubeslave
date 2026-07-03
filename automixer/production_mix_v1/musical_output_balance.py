"""Musical output balance solver for the Ayaic offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .models import amp_to_db, db_to_amp, ensure_stereo


MUSICAL_OUTPUT_BALANCE_STAGE = "musical_output_balance"
DEFAULT_MUSICAL_BALANCE_STYLE = "live_pop_rock"

RoleFn = Callable[[str], str]
AudioFn = Callable[[Any], np.ndarray]
LufsFn = Callable[[np.ndarray, int], float]
PeakFn = Callable[[np.ndarray], float]
BandEnergyFn = Callable[[np.ndarray, int], dict[str, float]]


@dataclass(frozen=True)
class BalanceMetrics:
    primary_lufs: float
    full_lufs: float
    peak_dbfs: float
    activity_ratio: float
    low_mid_db: float
    presence_db: float


ROLE_RULES: dict[str, dict[str, float | str]] = {
    "lead_vocal": {"relative_db": 0.0, "min_db": -3.0, "max_db": 3.0, "priority": 0, "bus": "foreground"},
    "backing_vocal": {"relative_db": -4.0, "min_db": -6.0, "max_db": 2.5, "priority": 3, "bus": "back_vocals"},
    "kick": {"relative_db": -6.0, "min_db": -4.5, "max_db": 2.0, "priority": 1, "bus": "foundation"},
    "snare": {"relative_db": -8.0, "min_db": -5.0, "max_db": 2.0, "priority": 2, "bus": "drums"},
    "tom": {"relative_db": -12.0, "min_db": -7.0, "max_db": 1.5, "priority": 4, "bus": "drums"},
    "overhead": {"relative_db": -14.0, "min_db": -9.0, "max_db": 3.5, "priority": 6, "bus": "image_air"},
    "room": {"relative_db": -19.0, "min_db": -10.0, "max_db": 0.5, "priority": 7, "bus": "image_air"},
    "bass": {"relative_db": -7.0, "min_db": -4.5, "max_db": 2.0, "priority": 1, "bus": "foundation"},
    "guitar": {"relative_db": -11.0, "min_db": -10.0, "max_db": 1.5, "priority": 5, "bus": "music_bed"},
    "keys": {"relative_db": -12.0, "min_db": -9.0, "max_db": 1.5, "priority": 5, "bus": "music_bed"},
    "pad": {"relative_db": -15.0, "min_db": -10.0, "max_db": 1.0, "priority": 7, "bus": "music_bed"},
    "playback": {"relative_db": -14.0, "min_db": -9.0, "max_db": 1.0, "priority": 6, "bus": "music_bed"},
    "unknown": {"relative_db": -16.0, "min_db": -2.5, "max_db": 1.0, "priority": 9, "bus": "unknown"},
}


STYLE_MODIFIERS: dict[str, dict[str, float]] = {
    "live_pop_rock": {"bed_down_db": 0.0, "vocal_up_db": 0.0, "foundation_up_db": 0.0},
    "rock": {"bed_down_db": 0.6, "vocal_up_db": -0.4, "foundation_up_db": 0.8},
    "dense": {"bed_down_db": 1.2, "vocal_up_db": 0.5, "foundation_up_db": 0.2},
    "ballad": {"bed_down_db": 0.8, "vocal_up_db": 0.8, "foundation_up_db": -0.6},
    "transparent": {"bed_down_db": 1.4, "vocal_up_db": 0.6, "foundation_up_db": -0.8},
}


def apply_musical_output_balance(
    stems: list[Any],
    *,
    style: str = DEFAULT_MUSICAL_BALANCE_STYLE,
    report_only: bool = False,
    role_fn: RoleFn | None = None,
    primary_audio_fn: AudioFn | None = None,
    lufs_fn: LufsFn | None = None,
    peak_fn: PeakFn | None = None,
    band_energy_fn: BandEnergyFn | None = None,
) -> dict[str, Any]:
    """Apply a musical output balance instead of final per-channel LUFS matching."""
    style_key = _style_key(style)
    if not stems:
        return {"enabled": False, "applied": False, "stage": MUSICAL_OUTPUT_BALANCE_STAGE, "reason": "no_stems"}

    sample_rate = int(stems[0].sample_rate)
    metrics_by_channel: dict[int, BalanceMetrics] = {}
    role_by_channel: dict[int, str] = {}
    for stem in stems:
        role = _normalize_role(role_fn(stem.name) if role_fn else _fallback_role(stem.name))
        role_by_channel[int(stem.channel_id)] = role
        metrics_by_channel[int(stem.channel_id)] = _measure_balance_metrics(
            stem,
            primary_audio_fn=primary_audio_fn,
            lufs_fn=lufs_fn,
            peak_fn=peak_fn,
            band_energy_fn=band_energy_fn,
        )

    section = _estimate_section(stems, role_by_channel, metrics_by_channel)
    density = _arrangement_density(metrics_by_channel)
    primary_channel = _select_primary_channel(stems, role_by_channel, metrics_by_channel)
    primary_metrics = metrics_by_channel[primary_channel] if primary_channel is not None else None
    primary_lufs = primary_metrics.primary_lufs if primary_metrics else _median_lufs(metrics_by_channel.values())

    raw_offsets = _raw_channel_offsets(
        stems,
        role_by_channel,
        metrics_by_channel,
        primary_lufs=primary_lufs,
        section=section,
        density=density,
        style=style_key,
    )
    linked_offsets = _link_stereo_offsets(stems, raw_offsets)
    bus_offsets = _bus_offsets(stems, role_by_channel, metrics_by_channel, linked_offsets, primary_lufs=primary_lufs)
    safe_offsets, blocked = _apply_safety_limits(stems, role_by_channel, metrics_by_channel, linked_offsets, bus_offsets)

    before_mix = _sum_stems(stems)
    before_mix_lufs = _call_lufs(lufs_fn, before_mix, sample_rate)
    before_mix_peak = _call_peak(peak_fn, before_mix)
    peak_safe_offsets, peak_blocks = _limit_mix_peak_growth(
        stems,
        safe_offsets,
        before_peak=before_mix_peak,
        peak_fn=peak_fn,
    )
    safe_offsets = peak_safe_offsets
    blocked.extend(peak_blocks)
    channel_reports = []
    for stem in stems:
        channel = int(stem.channel_id)
        role = role_by_channel[channel]
        metrics = metrics_by_channel[channel]
        offset = float(safe_offsets.get(channel, 0.0))
        if not report_only and abs(offset) > 1e-6:
            stem.audio = ensure_stereo(stem.audio) * db_to_amp(offset)
            stem.track_gain_db += offset
        report = {
            "stage": MUSICAL_OUTPUT_BALANCE_STAGE,
            "channel": channel,
            "file": Path(stem.path).name,
            "role": role,
            "bus": _role_rule(role)["bus"],
            "primary_source_channel": primary_channel,
            "primary_source_file": _file_for_channel(stems, primary_channel),
            "section": section["label"],
            "density_index": round(density, 4),
            "pre_primary_lufs": round(metrics.primary_lufs, 3),
            "pre_full_lufs": round(metrics.full_lufs, 3),
            "pre_peak_dbfs": round(metrics.peak_dbfs, 3),
            "requested_offset_db": round(raw_offsets.get(channel, 0.0), 3),
            "linked_offset_db": round(linked_offsets.get(channel, 0.0), 3),
            "bus_offset_db": round(bus_offsets.get(channel, 0.0), 3),
            "final_offset_db": round(offset, 3),
            "applied": bool(not report_only and abs(offset) >= 0.05),
            "report_only": bool(report_only),
            "reason": _balance_reason(role, offset, density, section["label"]),
        }
        _append_balance_note(stem, report)
        channel_reports.append(report)

    after_mix = _sum_stems(stems)
    after_mix_lufs = _call_lufs(lufs_fn, after_mix, sample_rate)
    after_mix_peak = _call_peak(peak_fn, after_mix)
    critic = _critic_report(
        stems,
        role_by_channel,
        metrics_by_channel,
        safe_offsets,
        before_peak=before_mix_peak,
        after_peak=after_mix_peak,
        primary_lufs=primary_lufs,
    )

    return {
        "enabled": True,
        "applied": any(abs(value) >= 0.05 for value in safe_offsets.values()) and not report_only,
        "stage": MUSICAL_OUTPUT_BALANCE_STAGE,
        "method": "mix_anchor_graph_output_balance_solver",
        "style": style_key,
        "section": section,
        "arrangement_density_index": round(density, 4),
        "arrangement_density_label": _density_label(density),
        "primary_source": {
            "channel": primary_channel,
            "file": _file_for_channel(stems, primary_channel),
            "role": role_by_channel.get(primary_channel, "unknown") if primary_channel is not None else "unknown",
            "primary_lufs": round(primary_lufs, 3),
        },
        "pre_mix_lufs": round(before_mix_lufs, 3),
        "post_mix_lufs": round(after_mix_lufs, 3),
        "pre_mix_peak_dbfs": round(before_mix_peak, 3),
        "post_mix_peak_dbfs": round(after_mix_peak, 3),
        "channel_offsets": channel_reports,
        "bus_offsets": _bus_report(stems, role_by_channel, metrics_by_channel, safe_offsets, lufs_fn=lufs_fn),
        "blocked_offsets": blocked,
        "critic": critic,
        "source": {
            "name": "Musical Output Balance",
            "source": "local mix-anchor graph and role-relative output solver",
            "references": [
                {
                    "name": "Automatic gain and fader control for live mixing",
                    "url": "https://www.eecs.qmul.ac.uk/~josh/documents/2009/PerezReiss-WASPAA2009-AutomaticGainandFaderControlForLiveMixing_001.pdf",
                    "use": "automatic fader control and safety-aware level adjustment",
                },
                {
                    "name": "Implementation and evaluation of autonomous multi-track fader control",
                    "url": "https://www.researchgate.net/publication/249649751_Implementation_and_Evaluation_of_Autonomous_Multi-track_Fader_Control",
                    "use": "relative fader balancing and multitrack output control",
                },
                {
                    "name": "Cross-adaptive processing as musical intervention",
                    "url": "https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00017/full",
                    "use": "cross-channel listening and musical interaction framing",
                },
            ],
            "notes": [
                "This stage replaces final channel/pair LUFS matching.",
                "Role/style targets are engineering heuristics; no strong evidence supports one universal balance table.",
            ],
        },
    }


def _measure_balance_metrics(
    stem: Any,
    *,
    primary_audio_fn: AudioFn | None,
    lufs_fn: LufsFn | None,
    peak_fn: PeakFn | None,
    band_energy_fn: BandEnergyFn | None,
) -> BalanceMetrics:
    sample_rate = int(stem.sample_rate)
    primary_audio = primary_audio_fn(stem) if primary_audio_fn else stem.audio
    band_energy = band_energy_fn(primary_audio, sample_rate) if band_energy_fn else {}
    activity_ratio = float(stem.bleed_analysis.get("post_phase_analysis_active_ratio") or stem.bleed_analysis.get("analysis_active_ratio") or 1.0)
    return BalanceMetrics(
        primary_lufs=_call_lufs(lufs_fn, primary_audio, sample_rate),
        full_lufs=_call_lufs(lufs_fn, stem.audio, sample_rate),
        peak_dbfs=_call_peak(peak_fn, stem.audio),
        activity_ratio=max(0.0, min(1.0, activity_ratio)),
        low_mid_db=float(band_energy.get("low_mid", -100.0)),
        presence_db=_presence_db(band_energy),
    )


def _raw_channel_offsets(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
    *,
    primary_lufs: float,
    section: dict[str, Any],
    density: float,
    style: str,
) -> dict[int, float]:
    offsets: dict[int, float] = {}
    for stem in stems:
        channel = int(stem.channel_id)
        role = role_by_channel[channel]
        metrics = metrics_by_channel[channel]
        rule = _role_rule(role)
        target_relative = float(rule["relative_db"])
        target_relative += _section_offset(role, section["label"])
        target_relative += _density_offset(role, density)
        target_relative += _style_offset(role, style)
        if role in {"guitar", "keys", "pad", "playback", "backing_vocal"}:
            target_relative += _masking_offset(metrics, primary_lufs)
        current_relative = metrics.primary_lufs - primary_lufs
        requested = target_relative - current_relative
        offsets[channel] = requested
    return offsets


def _link_stereo_offsets(stems: list[Any], raw_offsets: dict[int, float]) -> dict[int, float]:
    groups: dict[str, list[Any]] = {}
    for stem in stems:
        groups.setdefault(_link_key(stem.name, int(stem.channel_id)), []).append(stem)
    out = dict(raw_offsets)
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [float(raw_offsets.get(int(stem.channel_id), 0.0)) for stem in members]
        linked = float(np.median(values))
        for stem in members:
            out[int(stem.channel_id)] = linked
    return out


def _bus_offsets(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
    linked_offsets: dict[int, float],
    *,
    primary_lufs: float,
) -> dict[int, float]:
    out = {int(stem.channel_id): 0.0 for stem in stems}
    channels_by_bus: dict[str, list[int]] = {}
    for stem in stems:
        channel = int(stem.channel_id)
        bus = str(_role_rule(role_by_channel[channel])["bus"])
        channels_by_bus.setdefault(bus, []).append(channel)

    def bus_level(bus: str) -> float | None:
        values = [
            metrics_by_channel[channel].primary_lufs + float(linked_offsets.get(channel, 0.0))
            for channel in channels_by_bus.get(bus, [])
        ]
        return float(np.median(values)) if values else None

    music_bed = bus_level("music_bed")
    if music_bed is not None and music_bed > primary_lufs - 8.0:
        reduction = max(-3.0, (primary_lufs - 8.0) - music_bed)
        for channel in channels_by_bus.get("music_bed", []):
            out[channel] += reduction

    back_vocals = bus_level("back_vocals")
    if back_vocals is not None and back_vocals > primary_lufs - 3.0:
        reduction = max(-2.5, (primary_lufs - 3.0) - back_vocals)
        for channel in channels_by_bus.get("back_vocals", []):
            out[channel] += reduction

    foundation = bus_level("foundation")
    if foundation is not None and foundation > primary_lufs - 3.0:
        reduction = max(-2.0, (primary_lufs - 3.0) - foundation)
        for channel in channels_by_bus.get("foundation", []):
            out[channel] += reduction
    return out


def _apply_safety_limits(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
    linked_offsets: dict[int, float],
    bus_offsets: dict[int, float],
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    safe: dict[int, float] = {}
    blocked: list[dict[str, Any]] = []
    for stem in stems:
        channel = int(stem.channel_id)
        role = role_by_channel[channel]
        rule = _role_rule(role)
        requested = float(linked_offsets.get(channel, 0.0)) + float(bus_offsets.get(channel, 0.0))
        limited = max(float(rule["min_db"]), min(float(rule["max_db"]), requested))
        reason = []
        if abs(limited - requested) > 1e-6:
            reason.append("role_output_offset_limit")
        peak_limited = limited
        if limited > 0.0:
            peak_limit = -1.0 - metrics_by_channel[channel].peak_dbfs
            peak_limited = min(limited, max(0.0, peak_limit))
            if peak_limited < limited - 1e-6:
                reason.append("per_channel_peak_headroom_limited")
        if role == "unknown" and abs(peak_limited) > 2.5:
            peak_limited = max(-2.5, min(1.0, peak_limited))
            reason.append("unknown_channel_strict_limit")
        if abs(peak_limited) < 0.05:
            peak_limited = 0.0
        safe[channel] = peak_limited
        if reason:
            blocked.append(
                {
                    "channel": channel,
                    "file": Path(stem.path).name,
                    "role": role,
                    "requested_offset_db": round(requested, 3),
                    "safe_offset_db": round(peak_limited, 3),
                    "reasons": reason,
                }
            )
    return safe, blocked


def _limit_mix_peak_growth(
    stems: list[Any],
    offsets: dict[int, float],
    *,
    before_peak: float,
    peak_fn: PeakFn | None,
    max_growth_db: float = 1.5,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    safe = dict(offsets)
    blocked: list[dict[str, Any]] = []
    target_peak = before_peak + max_growth_db
    predicted = _sum_audio([
        ensure_stereo(stem.audio) * db_to_amp(float(safe.get(int(stem.channel_id), 0.0)))
        for stem in stems
    ])
    predicted_peak = _call_peak(peak_fn, predicted)
    if predicted_peak <= target_peak:
        return safe, blocked

    low = 0.0
    high = 1.0
    best_scale = 0.0
    best_peak = before_peak
    for _ in range(14):
        scale = (low + high) * 0.5
        candidate = _sum_audio([
            ensure_stereo(stem.audio) * db_to_amp(
                _scale_positive_offset(float(safe.get(int(stem.channel_id), 0.0)), scale)
            )
            for stem in stems
        ])
        candidate_peak = _call_peak(peak_fn, candidate)
        if candidate_peak <= target_peak:
            best_scale = scale
            best_peak = candidate_peak
            low = scale
        else:
            high = scale

    if best_scale >= 0.999:
        return safe, blocked

    for stem in stems:
        channel = int(stem.channel_id)
        previous = float(safe.get(channel, 0.0))
        scaled = _scale_positive_offset(previous, best_scale)
        if abs(scaled) < 0.05:
            scaled = 0.0
        safe[channel] = scaled
        if abs(previous - scaled) > 1e-3:
            blocked.append(
                {
                    "channel": channel,
                    "file": _file_for_channel(stems, channel),
                    "role": "mix_peak_guard",
                    "requested_offset_db": round(previous, 3),
                    "safe_offset_db": round(float(scaled), 3),
                    "reasons": ["mix_peak_growth_limited", "positive_offset_scale"],
                    "scale": round(float(best_scale), 4),
                    "predicted_peak_dbfs": round(predicted_peak, 3),
                    "safe_predicted_peak_dbfs": round(best_peak, 3),
                    "target_peak_dbfs": round(target_peak, 3),
                }
            )
    return safe, blocked


def _scale_positive_offset(offset_db: float, scale: float) -> float:
    """Scale only boosts during mix peak limiting, keeping corrective cuts intact."""

    if offset_db <= 0.0:
        return offset_db
    return offset_db * float(scale)


def _critic_report(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
    offsets: dict[int, float],
    *,
    before_peak: float,
    after_peak: float,
    primary_lufs: float,
) -> dict[str, Any]:
    notes = []
    if after_peak > before_peak + 0.25:
        notes.append({"severity": "medium", "reason": "mix_peak_increased", "delta_db": round(after_peak - before_peak, 3)})
    lead_channels = [int(stem.channel_id) for stem in stems if role_by_channel[int(stem.channel_id)] == "lead_vocal"]
    bed_channels = [
        int(stem.channel_id)
        for stem in stems
        if role_by_channel[int(stem.channel_id)] in {"guitar", "keys", "pad", "playback", "backing_vocal"}
    ]
    if lead_channels and bed_channels:
        lead_after = np.median([metrics_by_channel[ch].primary_lufs + offsets.get(ch, 0.0) for ch in lead_channels])
        bed_after = np.median([metrics_by_channel[ch].primary_lufs + offsets.get(ch, 0.0) for ch in bed_channels])
        if bed_after > lead_after - 4.0:
            notes.append(
                {
                    "severity": "medium",
                    "reason": "foreground_margin_still_small",
                    "lead_minus_bed_db": round(float(lead_after - bed_after), 3),
                }
            )
    big_boosts = [
        {
            "channel": int(stem.channel_id),
            "file": Path(stem.path).name,
            "offset_db": round(float(offsets.get(int(stem.channel_id), 0.0)), 3),
        }
        for stem in stems
        if offsets.get(int(stem.channel_id), 0.0) > 2.5
    ]
    if big_boosts:
        notes.append({"severity": "medium", "reason": "large_output_boosts", "channels": big_boosts})
    return {
        "decision": "ok" if not any(note["severity"] == "high" for note in notes) else "review",
        "notes": notes,
        "primary_lufs": round(primary_lufs, 3),
        "mix_peak_delta_db": round(after_peak - before_peak, 3),
    }


def _select_primary_channel(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
) -> int | None:
    active = [
        stem for stem in stems
        if metrics_by_channel[int(stem.channel_id)].activity_ratio > 0.01
    ]
    if not active:
        active = stems
    lead = [stem for stem in active if role_by_channel[int(stem.channel_id)] == "lead_vocal"]
    if lead:
        return max(lead, key=lambda stem: metrics_by_channel[int(stem.channel_id)].primary_lufs).channel_id
    foreground_roles = {"guitar", "keys", "playback"}
    candidates = [stem for stem in active if role_by_channel[int(stem.channel_id)] in foreground_roles]
    if candidates:
        return max(candidates, key=lambda stem: metrics_by_channel[int(stem.channel_id)].primary_lufs).channel_id
    if active:
        return min(active, key=lambda stem: float(_role_rule(role_by_channel[int(stem.channel_id)])["priority"])).channel_id
    return None


def _estimate_section(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
) -> dict[str, Any]:
    active_roles = {
        role_by_channel[int(stem.channel_id)]
        for stem in stems
        if metrics_by_channel[int(stem.channel_id)].activity_ratio > 0.01
    }
    density = _arrangement_density(metrics_by_channel)
    if "lead_vocal" in active_roles and "backing_vocal" in active_roles and density > 0.55:
        label = "chorus_or_pre_chorus"
        confidence = 0.68
        reason = "lead and backing vocals active in medium/dense arrangement"
    elif "lead_vocal" in active_roles:
        label = "verse"
        confidence = 0.62
        reason = "lead vocal is active"
    elif active_roles & {"guitar", "keys"}:
        label = "solo_or_instrumental"
        confidence = 0.55
        reason = "no lead vocal; melodic instruments active"
    else:
        label = "unknown"
        confidence = 0.35
        reason = "insufficient section evidence"
    return {"label": label, "confidence": confidence, "reason": reason}


def _arrangement_density(metrics_by_channel: dict[int, BalanceMetrics]) -> float:
    if not metrics_by_channel:
        return 0.0
    active = sum(1 for item in metrics_by_channel.values() if item.activity_ratio > 0.01)
    ratio = active / max(1, len(metrics_by_channel))
    loudness_values = [item.primary_lufs for item in metrics_by_channel.values() if item.primary_lufs > -80.0]
    loudness_density = 0.0
    if loudness_values:
        loudness_density = max(0.0, min(1.0, (np.percentile(loudness_values, 75) + 45.0) / 25.0))
    return float(max(0.0, min(1.0, 0.62 * ratio + 0.38 * loudness_density)))


def _density_label(density: float) -> str:
    if density < 0.25:
        return "sparse"
    if density < 0.45:
        return "light"
    if density < 0.70:
        return "medium"
    if density < 0.90:
        return "dense"
    return "very_dense"


def _section_offset(role: str, section: str) -> float:
    if section == "chorus_or_pre_chorus":
        if role == "backing_vocal":
            return 0.8
        if role in {"guitar", "keys", "playback"}:
            return 0.5
        if role in {"kick", "bass", "snare"}:
            return 0.4
    if section == "verse":
        if role in {"guitar", "keys", "pad", "playback"}:
            return -0.7
        if role == "lead_vocal":
            return 0.4
    if section == "solo_or_instrumental" and role in {"guitar", "keys"}:
        return 2.5
    return 0.0


def _density_offset(role: str, density: float) -> float:
    if density < 0.55:
        return 0.0
    amount = (density - 0.55) * 4.0
    if role == "overhead":
        return -amount * 0.35
    if role == "room":
        return -amount * 0.6
    if role in {"guitar", "keys", "pad", "playback"}:
        return -amount
    if role == "lead_vocal":
        return min(0.8, amount * 0.4)
    return 0.0


def _style_offset(role: str, style: str) -> float:
    mod = STYLE_MODIFIERS[_style_key(style)]
    if role == "overhead":
        return -float(mod["bed_down_db"]) * 0.35
    if role == "room":
        return -float(mod["bed_down_db"]) * 0.6
    if role in {"guitar", "keys", "pad", "playback"}:
        return -float(mod["bed_down_db"])
    if role == "lead_vocal":
        return float(mod["vocal_up_db"])
    if role in {"kick", "bass", "snare"}:
        return float(mod["foundation_up_db"])
    return 0.0


def _masking_offset(metrics: BalanceMetrics, primary_lufs: float) -> float:
    pressure = 0.0
    if metrics.presence_db > primary_lufs - 18.0:
        pressure -= min(2.0, (metrics.presence_db - (primary_lufs - 18.0)) * 0.12)
    if metrics.low_mid_db > primary_lufs - 12.0:
        pressure -= min(1.5, (metrics.low_mid_db - (primary_lufs - 12.0)) * 0.08)
    return pressure


def _bus_report(
    stems: list[Any],
    role_by_channel: dict[int, str],
    metrics_by_channel: dict[int, BalanceMetrics],
    offsets: dict[int, float],
    *,
    lufs_fn: LufsFn | None,
) -> list[dict[str, Any]]:
    by_bus: dict[str, list[Any]] = {}
    for stem in stems:
        bus = str(_role_rule(role_by_channel[int(stem.channel_id)])["bus"])
        by_bus.setdefault(bus, []).append(stem)
    report = []
    for bus, members in sorted(by_bus.items()):
        before_audio = _sum_audio([stem.audio / db_to_amp(offsets.get(int(stem.channel_id), 0.0)) for stem in members])
        after_audio = _sum_audio([stem.audio for stem in members])
        sample_rate = int(members[0].sample_rate)
        report.append(
            {
                "bus": bus,
                "members": [Path(stem.path).name for stem in members],
                "pre_lufs": round(_call_lufs(lufs_fn, before_audio, sample_rate), 3),
                "post_lufs": round(_call_lufs(lufs_fn, after_audio, sample_rate), 3),
                "median_offset_db": round(float(np.median([offsets.get(int(stem.channel_id), 0.0) for stem in members])), 3),
            }
        )
    return report


def _append_balance_note(stem: Any, report: dict[str, Any]) -> None:
    note = {
        "type": MUSICAL_OUTPUT_BALANCE_STAGE,
        "role": report["role"],
        "bus": report["bus"],
        "final_offset_db": report["final_offset_db"],
        "section": report["section"],
        "density_index": report["density_index"],
        "reason": report["reason"],
    }
    if hasattr(stem, "output_balance_notes"):
        stem.output_balance_notes.append(note)
    elif hasattr(stem, "notes"):
        stem.notes.append(note)


def _normalize_role(role: str) -> str:
    key = str(role or "unknown").strip().lower()
    if key in {"lead_vocal", "vocal", "vox"}:
        return "lead_vocal"
    if key in {"back_vocal", "backing_vocal", "backs"}:
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
    if "kick" in key:
        return "kick"
    if "snare" in key:
        return "snare"
    if "tom" in key:
        return "tom"
    if "bass" in key:
        return "bass"
    if "oh" in key or "overhead" in key:
        return "overhead"
    if "room" in key:
        return "room"
    if "back" in key or "backs" in key:
        return "backing_vocal"
    if "vox" in key or "vocal" in key:
        return "lead_vocal"
    if "guitar" in key or "gtr" in key:
        return "guitar"
    if "playback" in key:
        return "playback"
    if "accordion" in key or "keys" in key or "piano" in key:
        return "keys"
    return "unknown"


def _role_rule(role: str) -> dict[str, float | str]:
    return ROLE_RULES.get(role, ROLE_RULES["unknown"])


def _style_key(style: str) -> str:
    key = str(style or DEFAULT_MUSICAL_BALANCE_STYLE).strip().lower().replace("-", "_")
    return key if key in STYLE_MODIFIERS else DEFAULT_MUSICAL_BALANCE_STYLE


def _link_key(name: str, channel_id: int) -> str:
    key = " ".join(Path(str(name)).stem.lower().replace("_", " ").replace("-", " ").split())
    for suffix in (" l", " r", " left", " right"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return f"channel:{channel_id}"


def _file_for_channel(stems: list[Any], channel: int | None) -> str | None:
    if channel is None:
        return None
    for stem in stems:
        if int(stem.channel_id) == int(channel):
            return Path(stem.path).name
    return None


def _balance_reason(role: str, offset: float, density: float, section: str) -> str:
    direction = "held" if abs(offset) < 0.05 else "raised" if offset > 0.0 else "lowered"
    return f"{role} {direction} for {section}; density={density:.2f}; mix-anchor balance"


def _presence_db(band_energy: dict[str, float]) -> float:
    values = [float(band_energy.get(name, -100.0)) for name in ("high_mid", "high")]
    powers = [db_to_amp(value) ** 2 for value in values]
    return amp_to_db(float(np.sqrt(sum(powers) / max(1, len(powers)))))


def _median_lufs(values: Any) -> float:
    vals = [item.primary_lufs for item in values if item.primary_lufs > -100.0]
    return float(np.median(vals)) if vals else -30.0


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


def _call_lufs(lufs_fn: LufsFn | None, audio: np.ndarray, sample_rate: int) -> float:
    if lufs_fn:
        return float(lufs_fn(audio, sample_rate))
    return _rms_db(audio) - 0.691


def _call_peak(peak_fn: PeakFn | None, audio: np.ndarray) -> float:
    if peak_fn:
        return float(peak_fn(audio))
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.max(np.abs(arr))))


def _rms_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.sqrt(np.mean(arr * arr) + 1e-12)))
