"""Live shared-chat mix analysis for AutoFOH/WING soundcheck.

This module ports the previous offline folder-mixing workflow into a live,
bounded action planner. The core ideas are the same as
``tools/chat_only_shared_mix.py``:

- use the master spectrum as a balance meter;
- analyse +4.5 dB/oct compensated LTAS over the densest section;
- fix source/stem contributors rather than reaching for master EQ;
- build the pass around low-end, lead, rhythmic attack, music, and air anchors;
- keep every move small and safe for OSC application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from autofoh_safety import ChannelEQMove, ChannelFaderMove, HighPassAdjust, MasterFaderMove

try:
    from cross_adaptive_eq import CrossAdaptiveEQ
except Exception:
    CrossAdaptiveEQ = None


ANALYSIS_WINDOW_SEC = 8.0
COMPENSATION_DB_PER_OCTAVE = 4.5
EPS = 1e-12

BAND_SPECS: Dict[str, Tuple[float, float]] = {
    "50_100": (50.0, 100.0),
    "100_200": (100.0, 200.0),
    "200_500": (200.0, 500.0),
    "500_1000": (500.0, 1000.0),
    "1000_2500": (1000.0, 2500.0),
    "1500_4000": (1500.0, 4000.0),
    "2500_5000": (2500.0, 5000.0),
    "5000_8000": (5000.0, 8000.0),
    "6000_10000": (6000.0, 10000.0),
    "700_2000": (700.0, 2000.0),
    "8000_12000": (8000.0, 12000.0),
}

DISPLAY_CORRIDOR: Dict[str, Dict[str, float]] = {
    "50_100": {"min": 3.0, "max": 8.0, "target": 5.5},
    "100_200": {"min": 0.0, "max": 4.0, "target": 2.0},
    "200_500": {"min": -2.0, "max": 2.0, "target": 0.5},
    "500_1000": {"min": -1.5, "max": 1.0, "target": -0.2},
    "1000_2500": {"min": -1.0, "max": 1.0, "target": 0.0},
    "2500_5000": {"min": -2.0, "max": 0.5, "target": -0.8},
    "5000_8000": {"min": -4.0, "max": 0.0, "target": -1.8},
    "8000_12000": {"min": -6.0, "max": -1.0, "target": -3.0},
}

MIDLINE_BANDS = ("100_200", "200_500", "500_1000", "1000_2500", "2500_5000")


@dataclass
class LiveSharedMixConfig:
    enabled: bool = True
    analysis_window_sec: float = ANALYSIS_WINDOW_SEC
    max_actions_per_pass: int = 8
    master_peak_ceiling_db: float = -3.0
    master_max_cut_db: float = 1.0
    min_action_db: float = 0.25
    correct_master_output: bool = True
    mirror_eq_enabled: bool = True
    mirror_eq_max_actions_per_pass: int = 6
    mirror_eq_overlap_tolerance_db: float = 6.0
    mirror_eq_relative_floor_db: float = 24.0
    mirror_eq_max_cut_db: float = -3.0
    mirror_eq_max_boost_db: float = 1.5
    apply_routing_fixes: bool = False
    rename_generic_channels: bool = False

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]] = None) -> "LiveSharedMixConfig":
        payload = dict(payload or {})
        return cls(
            enabled=bool(payload.get("enabled", True)),
            analysis_window_sec=float(payload.get("analysis_window_sec", ANALYSIS_WINDOW_SEC)),
            max_actions_per_pass=int(payload.get("max_actions_per_pass", 8)),
            master_peak_ceiling_db=float(payload.get("master_peak_ceiling_db", -3.0)),
            master_max_cut_db=float(payload.get("master_max_cut_db", 1.0)),
            min_action_db=float(payload.get("min_action_db", 0.25)),
            correct_master_output=bool(payload.get("correct_master_output", True)),
            mirror_eq_enabled=bool(payload.get("mirror_eq_enabled", True)),
            mirror_eq_max_actions_per_pass=int(payload.get("mirror_eq_max_actions_per_pass", 6)),
            mirror_eq_overlap_tolerance_db=float(payload.get("mirror_eq_overlap_tolerance_db", 6.0)),
            mirror_eq_relative_floor_db=float(payload.get("mirror_eq_relative_floor_db", 24.0)),
            mirror_eq_max_cut_db=float(payload.get("mirror_eq_max_cut_db", -3.0)),
            mirror_eq_max_boost_db=float(payload.get("mirror_eq_max_boost_db", 1.5)),
            apply_routing_fixes=bool(payload.get("apply_routing_fixes", False)),
            rename_generic_channels=bool(payload.get("rename_generic_channels", False)),
        )


@dataclass
class LiveSharedMixChannel:
    channel_id: int
    name: str
    role: str
    stems: Tuple[str, ...]
    priority: float
    audio: np.ndarray
    sample_rate: int
    fader_db: float = -144.0
    muted: bool = False
    auto_corrections_enabled: bool = False
    raw_settings: Dict[str, Any] = field(default_factory=dict)
    current_eq_gain: Dict[int, float] = field(default_factory=dict)
    current_hpf_hz: float = 20.0
    hpf_enabled: bool = False


@dataclass
class LiveSharedMixPlan:
    actions: List[Any] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


def _amp_to_db(value: float) -> float:
    return float(20.0 * math.log10(max(float(value), EPS)))


def _db_to_amp(value: float) -> float:
    return float(10.0 ** (float(value) / 20.0))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 2:
        axis = 1 if data.shape[0] > data.shape[1] else 0
        return np.nan_to_num(np.mean(data, axis=axis).astype(np.float32))
    return np.nan_to_num(data.reshape(-1).astype(np.float32))


def _peak_db(audio: np.ndarray) -> float:
    data = np.asarray(audio, dtype=np.float32)
    if data.size == 0:
        return -120.0
    return _amp_to_db(float(np.max(np.abs(data))))


def _rms_db(audio: np.ndarray) -> float:
    data = _to_mono(audio)
    if data.size == 0:
        return -120.0
    return _amp_to_db(float(np.sqrt(np.mean(np.square(data))) + EPS))


def _analysis_window_start(audio: np.ndarray, sample_rate: int, window_sec: float) -> int:
    data = _to_mono(audio)
    window = max(4096, int(max(1.0, window_sec) * sample_rate))
    if data.size <= window:
        return 0
    hop = max(2048, window // 4)
    best_start = 0
    best_energy = -1.0
    for start in range(0, data.size - window + 1, hop):
        energy = float(np.mean(np.square(data[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start


def _ltas_spectrum(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    data = _to_mono(audio)
    if data.size <= 1:
        return np.array([0.0], dtype=np.float32), np.array([EPS], dtype=np.float32)
    n_fft = min(data.size, 16384)
    if n_fft < 1024:
        n_fft = data.size
    block = data[:n_fft]
    windowed = block * np.hanning(block.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(windowed)).astype(np.float32) + EPS
    freqs = np.fft.rfftfreq(block.size, 1.0 / sample_rate).astype(np.float32)
    return freqs, spec


def _compensated_band_levels(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    freqs, spec = _ltas_spectrum(audio, sample_rate)
    compensation_db = COMPENSATION_DB_PER_OCTAVE * np.log2(np.maximum(freqs, 1.0) / 100.0)
    weighted = spec * np.power(10.0, compensation_db / 20.0)
    levels: Dict[str, float] = {}
    for band_name, (low_hz, high_hz) in BAND_SPECS.items():
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(mask):
            levels[band_name] = -120.0
        else:
            levels[band_name] = _amp_to_db(float(np.mean(weighted[mask])))
    reference = float(np.median([levels[name] for name in MIDLINE_BANDS]))
    return {
        band_name: float(value - reference)
        for band_name, value in levels.items()
        if band_name in DISPLAY_CORRIDOR or band_name in {"1500_4000", "6000_10000", "700_2000"}
    }


def _raw_band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    freqs, spec = _ltas_spectrum(audio, sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask]))


def _segment(channel: LiveSharedMixChannel, start: int, end: int) -> np.ndarray:
    return _to_mono(channel.audio)[start:end].astype(np.float32)


def _analysis_mix(channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> np.ndarray:
    if end <= start:
        return np.zeros(0, dtype=np.float32)
    mix = np.zeros(end - start, dtype=np.float32)
    for channel in channels:
        if channel.muted:
            continue
        data = _segment(channel, start, end)
        if data.size < mix.size:
            data = np.pad(data, (0, mix.size - data.size))
        fader_db = float(channel.fader_db)
        if not np.isfinite(fader_db):
            fader_db = -144.0
        mix += data[:mix.size] * _db_to_amp(max(-144.0, min(10.0, fader_db)))
    return mix.astype(np.float32)


def _band_summary(channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> Dict[str, float]:
    if not channels:
        return {band: -120.0 for band in DISPLAY_CORRIDOR}
    levels = _compensated_band_levels(_analysis_mix(channels, start, end), channels[0].sample_rate)
    return {band: round(float(levels.get(band, -120.0)), 2) for band in DISPLAY_CORRIDOR}


def _band_shares(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    low_hz: float,
    high_hz: float,
) -> Dict[int, float]:
    energies: Dict[int, float] = {}
    total = 0.0
    for channel in channels:
        if channel.muted:
            energies[channel.channel_id] = 0.0
            continue
        energy = _raw_band_energy(_segment(channel, start, end), channel.sample_rate, low_hz, high_hz)
        energies[channel.channel_id] = energy
        total += energy
    if total <= EPS:
        return {channel_id: 0.0 for channel_id in energies}
    return {channel_id: float(value / total) for channel_id, value in energies.items()}


def _top_culprits(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    low_hz: float,
    high_hz: float,
    *,
    allowed_roles: Optional[set[str]] = None,
    exclude_roles: Optional[set[str]] = None,
    limit: int = 3,
) -> List[Tuple[LiveSharedMixChannel, float]]:
    shares = _band_shares(channels, start, end, low_hz, high_hz)
    result: List[Tuple[LiveSharedMixChannel, float]] = []
    for channel in channels:
        if allowed_roles is not None and channel.role not in allowed_roles:
            continue
        if exclude_roles is not None and channel.role in exclude_roles:
            continue
        result.append((channel, shares.get(channel.channel_id, 0.0)))
    result.sort(key=lambda item: item[1], reverse=True)
    return result[:limit]


def _lead_channels(channels: Sequence[LiveSharedMixChannel]) -> List[LiveSharedMixChannel]:
    return [channel for channel in channels if channel.role == "lead_vocal"]


def _lead_masking_state(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
) -> Dict[str, float]:
    lead_energy = 0.0
    accompaniment_energy = 0.0
    lead_bright = 0.0
    total_bright = 0.0
    for channel in channels:
        if channel.muted:
            continue
        seg = _segment(channel, start, end)
        energy = _raw_band_energy(seg, channel.sample_rate, 1500.0, 4000.0)
        if channel.role == "lead_vocal":
            lead_energy += energy
        else:
            accompaniment_energy += energy
        bright = _raw_band_energy(seg, channel.sample_rate, 6000.0, 10000.0)
        total_bright += bright
        if channel.role == "lead_vocal":
            lead_bright += bright
    total = lead_energy + accompaniment_energy + EPS
    return {
        "lead_share_1500_4000": float(lead_energy / total),
        "accompaniment_share_1500_4000": float(accompaniment_energy / total),
        "lead_sibilance_share_6000_10000": float(lead_bright / (total_bright + EPS)),
    }


def _eq_band_for_freq(freq_hz: float) -> int:
    if freq_hz < 220.0:
        return 1
    if freq_hz < 1200.0:
        return 2
    if freq_hz < 5500.0:
        return 3
    return 4


def _append_eq_delta(
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    channel: LiveSharedMixChannel,
    freq_hz: float,
    delta_db: float,
    q: float,
    reason: str,
    config: LiveSharedMixConfig,
) -> None:
    if not channel.auto_corrections_enabled or channel.muted:
        decisions.append(_skip(channel, reason, "channel muted or not eligible"))
        return
    delta = float(np.clip(delta_db, -1.0, 1.0))
    if abs(delta) < config.min_action_db:
        return
    band = _eq_band_for_freq(freq_hz)
    current = float(channel.current_eq_gain.get(band, 0.0))
    target = round(current + delta, 2)
    channel.current_eq_gain[band] = target
    action = ChannelEQMove(
        channel_id=channel.channel_id,
        band=band,
        freq_hz=float(freq_hz),
        gain_db=target,
        q=float(q),
        reason=reason,
    )
    actions.append(action)
    decisions.append(_decision(channel, "eq", reason, target_db=target, freq_hz=freq_hz, q=q))


def _append_fader_delta(
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    channel: LiveSharedMixChannel,
    delta_db: float,
    reason: str,
    config: LiveSharedMixConfig,
) -> None:
    if not channel.auto_corrections_enabled or channel.muted:
        decisions.append(_skip(channel, reason, "channel muted or not eligible"))
        return
    delta = float(np.clip(delta_db, -1.0, 1.0))
    if abs(delta) < config.min_action_db:
        return
    target = max(-144.0, min(0.0, channel.fader_db + delta))
    if abs(target - channel.fader_db) < config.min_action_db:
        return
    channel.fader_db = target
    action = ChannelFaderMove(
        channel_id=channel.channel_id,
        target_db=round(target, 2),
        is_lead=channel.role == "lead_vocal",
        reason=reason,
    )
    actions.append(action)
    decisions.append(_decision(channel, "fader", reason, target_db=target, delta_db=delta))


def _append_hpf_if_higher(
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    channel: LiveSharedMixChannel,
    freq_hz: float,
    reason: str,
    config: LiveSharedMixConfig,
) -> None:
    if not channel.auto_corrections_enabled or channel.muted:
        decisions.append(_skip(channel, reason, "channel muted or not eligible"))
        return
    if freq_hz <= channel.current_hpf_hz + 5.0:
        return
    action = HighPassAdjust(
        channel_id=channel.channel_id,
        freq_hz=float(freq_hz),
        enabled=True,
        reason=reason,
    )
    channel.current_hpf_hz = float(freq_hz)
    actions.append(action)
    decisions.append(_decision(channel, "hpf", reason, target_hz=freq_hz))


def _decision(channel: LiveSharedMixChannel, action: str, reason: str, **payload: Any) -> Dict[str, Any]:
    return {
        "channel": int(channel.channel_id),
        "name": channel.name,
        "role": channel.role,
        "action": action,
        "reason": reason,
        **payload,
    }


def _skip(channel: LiveSharedMixChannel, reason: str, skip_reason: str) -> Dict[str, Any]:
    return {
        "channel": int(channel.channel_id),
        "name": channel.name,
        "role": channel.role,
        "action": "skip",
        "reason": reason,
        "skip_reason": skip_reason,
    }


def _phase_log(
    label: str,
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
) -> Dict[str, Any]:
    mask = _lead_masking_state(channels, start, end)
    return {
        "phase": label,
        "band_deviation_db": _band_summary(channels, start, end),
        "lead_masking": {key: round(float(value), 3) for key, value in mask.items()},
    }


def _names_for_roles(channels: Sequence[LiveSharedMixChannel], roles: set[str]) -> set[int]:
    return {channel.channel_id for channel in channels if channel.role in roles}


def _filter_channels(channels: Sequence[LiveSharedMixChannel], ids: set[int]) -> List[LiveSharedMixChannel]:
    return [channel for channel in channels if channel.channel_id in ids]


def _apply_low_end_anchor_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    config: LiveSharedMixConfig,
) -> None:
    summary = _band_summary(channels, start, end)
    if summary["50_100"] > DISPLAY_CORRIDOR["50_100"]["max"]:
        for channel, share in _top_culprits(channels, start, end, 50.0, 100.0, allowed_roles={"kick", "bass"}, limit=2):
            if share < 0.25:
                continue
            freq = 78.0 if channel.role == "kick" else 82.0
            _append_eq_delta(actions, decisions, channel, freq, -1.0, 1.0, "50-100 Hz overweight: split kick/bass roles", config)
            _append_fader_delta(actions, decisions, channel, -0.5, "50-100 Hz overweight: keep low-end anchor controlled", config)
            break
    if summary["100_200"] > DISPLAY_CORRIDOR["100_200"]["max"]:
        for channel, share in _top_culprits(channels, start, end, 100.0, 200.0, allowed_roles={"kick", "bass", "toms", "guitars"}, limit=3):
            if share >= 0.18:
                _append_eq_delta(actions, decisions, channel, 150.0, -1.0, 0.9, "100-200 Hz body excess on low-end anchor phase", config)
                break
    bass = next((channel for channel in channels if channel.role == "bass"), None)
    if bass is not None:
        bass_low = _raw_band_energy(_segment(bass, start, end), bass.sample_rate, 60.0, 120.0)
        bass_audibility = _raw_band_energy(_segment(bass, start, end), bass.sample_rate, 700.0, 2000.0)
        if bass_low > EPS and (bass_audibility / bass_low) < 0.14:
            _append_eq_delta(actions, decisions, bass, 1200.0, 1.0, 1.0, "Bass audibility support from 700 Hz - 2 kHz rule", config)


def _apply_lead_anchor_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    config: LiveSharedMixConfig,
) -> None:
    leads = _lead_channels(channels)
    if len(leads) >= 2:
        levels = {channel.channel_id: _rms_db(_segment(channel, start, end)) for channel in leads}
        target = float(np.median(list(levels.values())))
        for channel in leads:
            shortfall = target - levels[channel.channel_id]
            if shortfall > 0.9:
                _append_fader_delta(channel=channel, actions=actions, decisions=decisions, delta_db=min(1.0, shortfall), reason="Lead layer parity from vocal anchor logic", config=config)

    summary = _band_summary(channels, start, end)
    mask = _lead_masking_state(channels, start, end)
    if mask["lead_share_1500_4000"] < 0.34:
        for channel, share in _top_culprits(
            channels,
            start,
            end,
            1500.0,
            4000.0,
            allowed_roles={"guitars", "playback", "bgv", "snare", "hi_hat", "ride", "overheads_room"},
            limit=3,
        ):
            if share < 0.10:
                continue
            freq = 2500.0 if channel.role in {"guitars", "playback", "bgv"} else 3200.0
            _append_eq_delta(actions, decisions, channel, freq, -1.0, 1.0, "Free 1.5-4 kHz space around lead instead of master EQ", config)
        if mask["lead_share_1500_4000"] < 0.28:
            for channel in leads:
                _append_fader_delta(actions, decisions, channel, 0.5, "Small lead support after competitor EQ in 1.5-4 kHz", config)
    if summary["1000_2500"] < DISPLAY_CORRIDOR["1000_2500"]["min"]:
        for channel in leads:
            _append_eq_delta(actions, decisions, channel, 2200.0, 0.8, 1.0, "Lead support when 1-2.5 kHz drops below corridor", config)
            _append_fader_delta(actions, decisions, channel, 0.5, "Small lead anchor lift for underfilled 1-2.5 kHz", config)


def _apply_rhythm_anchor_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    config: LiveSharedMixConfig,
) -> None:
    summary = _band_summary(channels, start, end)
    if summary["2500_5000"] > DISPLAY_CORRIDOR["2500_5000"]["max"]:
        snare = next((channel for channel in channels if channel.role == "snare"), None)
        if snare is not None:
            _append_eq_delta(actions, decisions, snare, 3500.0, -0.8, 1.2, "2.5-5 kHz aggression after adding rhythmic attack", config)
    if summary["100_200"] > DISPLAY_CORRIDOR["100_200"]["max"]:
        for channel in channels:
            if channel.role == "toms":
                _append_eq_delta(actions, decisions, channel, 180.0, -0.8, 1.0, "Tom body excess in rhythmic anchor phase", config)


def _apply_music_layer_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    config: LiveSharedMixConfig,
) -> None:
    summary = _band_summary(channels, start, end)
    if summary["200_500"] > DISPLAY_CORRIDOR["200_500"]["max"]:
        for channel, share in _top_culprits(
            channels,
            start,
            end,
            200.0,
            500.0,
            allowed_roles={"guitars", "playback", "bgv", "overheads_room", "lead_vocal"},
            exclude_roles={"kick", "bass"},
            limit=4,
        ):
            if share < 0.12:
                continue
            _append_eq_delta(actions, decisions, channel, 320.0, -1.0, 0.9, "200-500 Hz blanket: clean culprit stem, not master EQ", config)
            if channel.role in {"guitars", "playback", "bgv", "overheads_room"}:
                _append_hpf_if_higher(actions, decisions, channel, 120.0 if channel.role in {"guitars", "playback"} else 150.0, "Secondary layer HPF from 200-500 Hz buildup rule", config)
    _apply_lead_anchor_rules(channels, start, end, actions, decisions, config)


def _apply_cymbal_air_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    config: LiveSharedMixConfig,
) -> None:
    summary = _band_summary(channels, start, end)
    if not (
        summary["5000_8000"] > DISPLAY_CORRIDOR["5000_8000"]["max"]
        or summary["8000_12000"] > DISPLAY_CORRIDOR["8000_12000"]["max"]
    ):
        return
    for channel, share in _top_culprits(
        channels,
        start,
        end,
        6000.0,
        10000.0,
        allowed_roles={"hi_hat", "ride", "overheads_room", "bgv", "playback", "lead_vocal"},
        limit=5,
    ):
        if share < 0.08:
            continue
        if channel.role == "hi_hat":
            _append_eq_delta(actions, decisions, channel, 7800.0, -1.0, 1.4, "Hi-hat must not dominate 6-10 kHz brightness", config)
            _append_fader_delta(actions, decisions, channel, -0.5, "Reduce hi-hat dominance in brightness band", config)
        elif channel.role == "ride":
            _append_eq_delta(actions, decisions, channel, 6800.0, -1.0, 1.3, "Ride dominance in 6-10 kHz", config)
            _append_fader_delta(actions, decisions, channel, -0.4, "Reduce ride dominance in brightness band", config)
        elif channel.role == "overheads_room":
            _append_eq_delta(actions, decisions, channel, 7600.0, -1.0, 1.0, "Overheads/room should not build a 6-10 kHz plate", config)
            _append_hpf_if_higher(actions, decisions, channel, 180.0, "Keep overhead/room low-mid from building a 200-500 Hz blanket", config)
        elif channel.role in {"bgv", "lead_vocal"}:
            _append_eq_delta(actions, decisions, channel, 7200.0, -0.6, 1.6, "Tame vocal sibilance only if it dominates 6-10 kHz", config)
        elif channel.role == "playback":
            _append_eq_delta(actions, decisions, channel, 7000.0, -0.8, 1.1, "Playback brightness must leave room for cymbals and lead", config)


MIRROR_EQ_BAND_RANGES: Dict[str, Tuple[float, float]] = {
    "sub": (35.0, 70.0),
    "bass": (70.0, 160.0),
    "low_mid": (180.0, 500.0),
    "mid": (700.0, 2000.0),
    "high_mid": (2000.0, 5000.0),
    "high": (5000.0, 9000.0),
    "air": (9000.0, 14000.0),
}


def _mirror_eq_band_energy(
    channel: LiveSharedMixChannel,
    start: int,
    end: int,
    relative_floor_db: float = 24.0,
) -> Dict[str, float]:
    audio = _segment(channel, start, end)
    levels = {
        band: _amp_to_db(_raw_band_energy(audio, channel.sample_rate, low_hz, high_hz) + EPS)
        for band, (low_hz, high_hz) in MIRROR_EQ_BAND_RANGES.items()
    }
    if not levels:
        return {}
    peak_level = max(levels.values())
    floor = peak_level - max(0.0, float(relative_floor_db))
    return {
        band: value if value >= floor else -120.0
        for band, value in levels.items()
    }


def _append_mirror_eq_rules(
    channels: Sequence[LiveSharedMixChannel],
    start: int,
    end: int,
    actions: List[Any],
    decisions: List[Dict[str, Any]],
    report: Dict[str, Any],
    config: LiveSharedMixConfig,
) -> None:
    """Apply full mirror EQ: cut masker, smaller boost on masked source."""
    if not config.mirror_eq_enabled:
        report["mirror_eq"] = {"enabled": False, "reason": "disabled"}
        return
    if CrossAdaptiveEQ is None:
        report["mirror_eq"] = {"enabled": False, "reason": "cross_adaptive_eq_unavailable"}
        return

    eligible = [
        channel
        for channel in channels
        if channel.auto_corrections_enabled
        and not channel.muted
        and channel.fader_db > -90.0
        and _peak_db(channel.audio) > -65.0
    ]
    if len(eligible) < 2:
        report["mirror_eq"] = {"enabled": True, "reason": "not_enough_eligible_channels"}
        return

    channel_band_energy = {
        channel.channel_id: _mirror_eq_band_energy(
            channel,
            start,
            end,
            config.mirror_eq_relative_floor_db,
        )
        for channel in eligible
    }
    channel_priorities = {
        channel.channel_id: -float(channel.priority)
        for channel in eligible
    }
    mirror = CrossAdaptiveEQ(
        overlap_tolerance_db=float(config.mirror_eq_overlap_tolerance_db),
        max_cut_db=-abs(float(config.mirror_eq_max_cut_db)),
        max_boost_db=max(0.0, float(config.mirror_eq_max_boost_db)),
    )
    adjustments = mirror.calculate_corrections(channel_band_energy, channel_priorities)

    by_key: Dict[Tuple[int, int], Any] = {}
    for adjustment in adjustments:
        band = _eq_band_for_freq(float(adjustment.frequency_hz))
        key = (int(adjustment.channel_id), int(band))
        previous = by_key.get(key)
        if previous is None or abs(float(adjustment.gain_db)) > abs(float(previous.gain_db)):
            by_key[key] = adjustment

    cuts = sorted(
        (item for item in by_key.values() if float(item.gain_db) < 0.0),
        key=lambda item: -abs(float(item.gain_db)),
    )
    boosts = sorted(
        (item for item in by_key.values() if float(item.gain_db) > 0.0),
        key=lambda item: -abs(float(item.gain_db)),
    )
    selected = []
    limit = max(0, int(config.mirror_eq_max_actions_per_pass))
    while len(selected) < limit and (cuts or boosts):
        if cuts and len(selected) < limit:
            selected.append(cuts.pop(0))
        if boosts and len(selected) < limit:
            selected.append(boosts.pop(0))

    channel_by_id = {channel.channel_id: channel for channel in eligible}
    sent_candidates = []
    for adjustment in selected:
        channel = channel_by_id.get(int(adjustment.channel_id))
        if channel is None:
            continue
        direction = "cut masker" if float(adjustment.gain_db) < 0.0 else "boost masked source"
        reason = (
            f"Mirror EQ {direction}: cross-adaptive overlap at "
            f"{float(adjustment.frequency_hz):.0f}Hz"
        )
        before_count = len(actions)
        _append_eq_delta(
            actions,
            decisions,
            channel,
            float(adjustment.frequency_hz),
            float(adjustment.gain_db),
            float(adjustment.q_factor),
            reason,
            config,
        )
        if len(actions) > before_count:
            sent_candidates.append(
                {
                    "channel": int(channel.channel_id),
                    "name": channel.name,
                    "role": channel.role,
                    "freq_hz": round(float(adjustment.frequency_hz), 1),
                    "delta_db": round(float(np.clip(adjustment.gain_db, -1.0, 1.0)), 2),
                    "q": round(float(adjustment.q_factor), 2),
                    "direction": direction,
                }
            )

    report["mirror_eq"] = {
        "enabled": True,
        "mode": "cross_adaptive_full_mirror_eq",
        "candidate_count": len(adjustments),
        "planned_candidates": sent_candidates,
        "principle": "cut lower-priority masker with narrower Q, boost higher-priority masked source more gently",
    }


def _append_master_action(
    actions: List[Any],
    report: Dict[str, Any],
    master_audio: Optional[np.ndarray],
    master_current_fader_db: Optional[float],
    config: LiveSharedMixConfig,
) -> None:
    if master_audio is None or not config.correct_master_output:
        report["master"] = {"enabled": False, "reason": "no_master_reference_audio"}
        return
    peak_db = _peak_db(master_audio)
    rms = _rms_db(master_audio)
    report["master"] = {
        "enabled": True,
        "peak_dbfs": round(float(peak_db), 2),
        "rms_db": round(float(rms), 2),
        "peak_ceiling_dbfs": round(float(config.master_peak_ceiling_db), 2),
        "principle": "master spectrum is a balance meter; source/stem fixes are preferred",
    }
    excess = peak_db - float(config.master_peak_ceiling_db)
    if excess <= config.min_action_db:
        report["master"]["action"] = "none"
        report["master"]["reason"] = "master peak inside live ceiling"
        return
    if master_current_fader_db is None:
        report["master"]["action"] = "skip"
        report["master"]["reason"] = "main fader readback unavailable"
        return
    cut = min(float(config.master_max_cut_db), excess)
    target = min(0.0, float(master_current_fader_db) - cut)
    actions.append(
        MasterFaderMove(
            main_id=1,
            target_db=round(target, 2),
            reason="Master reference peak exceeds live ceiling; reduce Main 1 only",
        )
    )
    report["master"].update({
        "action": "master_fader_cut",
        "current_fader_db": round(float(master_current_fader_db), 2),
        "target_fader_db": round(float(target), 2),
        "cut_db": round(float(cut), 2),
    })


def _action_plan_priority(action: Any) -> int:
    if isinstance(action, MasterFaderMove):
        return 0
    if isinstance(action, ChannelEQMove) and str(getattr(action, "reason", "")).startswith("Mirror EQ"):
        return 1
    if isinstance(action, ChannelFaderMove):
        return 2
    if isinstance(action, ChannelEQMove):
        return 3
    if isinstance(action, HighPassAdjust):
        return 4
    return 5


def _limit_actions(actions: Sequence[Any], limit: int) -> List[Any]:
    max_actions = max(0, int(limit))
    if len(actions) <= max_actions:
        return list(actions)
    ranked = sorted(
        enumerate(actions),
        key=lambda item: (_action_plan_priority(item[1]), item[0]),
    )
    return [action for _, action in ranked[:max_actions]]


def build_live_shared_mix_plan(
    channels: Sequence[LiveSharedMixChannel],
    sample_rate: int,
    *,
    config: Optional[LiveSharedMixConfig] = None,
    master_audio: Optional[np.ndarray] = None,
    master_current_fader_db: Optional[float] = None,
) -> LiveSharedMixPlan:
    """Build a bounded live OSC action plan from current channel audio buffers."""

    config = config or LiveSharedMixConfig()
    if not config.enabled:
        return LiveSharedMixPlan(report={"enabled": False, "reason": "disabled"})

    active = [
        channel
        for channel in channels
        if channel.audio.size > 0
        and not channel.muted
        and channel.fader_db > -90.0
        and _peak_db(channel.audio) > -65.0
    ]
    actions: List[Any] = []
    decisions: List[Dict[str, Any]] = []
    if not active:
        return LiveSharedMixPlan(report={"enabled": config.enabled, "reason": "no_active_channels"})

    rough = _analysis_mix(active, 0, min(len(_to_mono(active[0].audio)), int(config.analysis_window_sec * sample_rate)))
    start = _analysis_window_start(rough, sample_rate, config.analysis_window_sec)
    end = start + min(
        int(max(1.0, config.analysis_window_sec) * sample_rate),
        min(_to_mono(channel.audio).size for channel in active),
    )
    if end <= start:
        start = 0
        end = min(_to_mono(channel.audio).size for channel in active)

    report: Dict[str, Any] = {
        "enabled": bool(config.enabled),
        "mode": "live_shared_chat_mix",
        "analysis_window_sec": round((end - start) / float(sample_rate), 2),
        "analysis_window_start_sec": round(start / float(sample_rate), 2),
        "rules": [
            "master spectrum used as balance meter only",
            "+4.5 dB/oct compensated LTAS",
            "anchor order: kick+bass -> lead -> rhythmic attack -> music -> cymbals/air",
            "source/stem fixes before master processing",
            "vocal space created by EQ on competitors",
            "mirror EQ: cut masker and gently boost masked priority source in overlapping bands",
            "small bounded fader/EQ/HPF moves only",
        ],
        "analysis_before": _phase_log("before", active, start, end),
        "phases": [],
        "decisions": decisions,
        "routing_audit": _routing_audit(active, config),
    }

    low_end_ids = _names_for_roles(active, {"kick", "bass"})
    low_end = _filter_channels(active, low_end_ids)
    _apply_low_end_anchor_rules(low_end, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("kick_bass_anchor", active, start, end))

    lead_ids = low_end_ids | _names_for_roles(active, {"lead_vocal"})
    lead_layer = _filter_channels(active, lead_ids)
    _apply_lead_anchor_rules(lead_layer, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("lead_anchor", active, start, end))

    rhythm_ids = lead_ids | _names_for_roles(active, {"snare", "toms"})
    rhythm = _filter_channels(active, rhythm_ids)
    _apply_rhythm_anchor_rules(rhythm, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("rhythm_anchor", active, start, end))

    music_ids = rhythm_ids | _names_for_roles(active, {"guitars", "playback", "bgv"})
    music = _filter_channels(active, music_ids)
    _apply_music_layer_rules(music, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("music_layer", active, start, end))

    cymbal_ids = music_ids | _names_for_roles(active, {"hi_hat", "ride", "overheads_room"})
    cymbals = _filter_channels(active, cymbal_ids)
    _apply_cymbal_air_rules(cymbals, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("cymbal_air_layer", active, start, end))

    _append_mirror_eq_rules(active, start, end, actions, decisions, report, config)

    report["analysis_after"] = _phase_log("after", active, start, end)
    _append_master_action(actions, report, master_audio, master_current_fader_db, config)

    limited_actions = _limit_actions(actions, int(config.max_actions_per_pass))
    report["actions_requested"] = len(actions)
    report["actions_planned"] = len(limited_actions)
    report["actions_truncated"] = max(0, len(actions) - len(limited_actions))
    report["planned_action_types"] = [action.action_type for action in limited_actions]
    return LiveSharedMixPlan(actions=limited_actions, report=report)


def _routing_audit(
    channels: Sequence[LiveSharedMixChannel],
    config: LiveSharedMixConfig,
) -> List[Dict[str, Any]]:
    audit = []
    for channel in channels:
        raw = channel.raw_settings
        route = raw.get("input_routing") or {}
        main_send = raw.get("main_send") or {}
        name_generic = channel.name.strip().upper() in {
            f"CH{channel.channel_id}",
            f"CH {channel.channel_id}",
            "",
        }
        item = {
            "channel": int(channel.channel_id),
            "name": channel.name,
            "role": channel.role,
            "input_routing": route,
            "main_send": main_send,
            "name_generic": bool(name_generic),
            "routing_write_enabled": bool(config.apply_routing_fixes),
            "rename_enabled": bool(config.rename_generic_channels),
        }
        if not config.apply_routing_fixes:
            item["routing_decision"] = "observe_only_no_patch_map"
        if name_generic and not config.rename_generic_channels:
            item["name_decision"] = "observe_only_generic_name"
        audit.append(item)
    return audit


def normalize_live_role(preset: str = "", source_role: str = "", name: str = "") -> str:
    text = " ".join([str(preset or ""), str(source_role or ""), str(name or "")]).lower()
    if "kick" in text:
        return "kick"
    if "snare" in text or " sn " in f" {text} ":
        return "snare"
    if "tom" in text:
        return "toms"
    if "hat" in text or "hihat" in text:
        return "hi_hat"
    if "ride" in text:
        return "ride"
    if "overhead" in text or "room" in text or "ohl" in text or "ohr" in text:
        return "overheads_room"
    if "bass" in text:
        return "bass"
    if "guitar" in text or "gtr" in text:
        return "guitars"
    if "lead" in text and ("vox" in text or "vocal" in text):
        return "lead_vocal"
    if "vox" in text or "vocal" in text or "backs" in text or "bgv" in text:
        return "bgv"
    if "playback" in text or "tracks" in text or "pb" in text:
        return "playback"
    if "accordion" in text or "keys" in text or "synth" in text:
        return "playback"
    return "unknown"
