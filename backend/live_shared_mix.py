"""Pure live shared-mix planner for AutoFOH.

The planner never talks to a console. It converts current audio buffers and
readback state into bounded typed actions consumed by AutoFOHSafetyController.
Analysis is performed on the audible, fader-weighted signal over the selected
programme window rather than on the first FFT frame of raw inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from autofoh_safety import ChannelEQMove, ChannelFaderMove, HighPassAdjust, MasterFaderMove

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
        p = dict(payload or {})
        defaults = cls()
        values: Dict[str, Any] = {}
        for name in cls.__dataclass_fields__:
            value = p.get(name, getattr(defaults, name))
            expected = type(getattr(defaults, name))
            try:
                values[name] = expected(value)
            except (TypeError, ValueError):
                values[name] = getattr(defaults, name)
        return cls(**values)


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
    if not np.isfinite(value):
        return 0.0
    return float(10.0 ** (float(np.clip(value, -144.0, 10.0)) / 20.0))


def _to_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        # Audio in the project appears in both sample-major and channel-major form.
        channel_axis = 1 if data.shape[0] >= data.shape[1] else 0
        return np.mean(data, axis=channel_axis, dtype=np.float64).astype(np.float32)
    return data.reshape(-1).astype(np.float32)


def _peak_db(audio: np.ndarray) -> float:
    data = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return -120.0 if data.size == 0 else _amp_to_db(float(np.max(np.abs(data))))


def _rms_db(audio: np.ndarray) -> float:
    data = _to_mono(audio)
    if data.size == 0:
        return -120.0
    return _amp_to_db(float(np.sqrt(np.mean(data.astype(np.float64) ** 2)) + EPS))


def _analysis_window_start(audio: np.ndarray, sample_rate: int, window_sec: float) -> int:
    data = _to_mono(audio)
    window = max(4096, int(max(1.0, float(window_sec)) * sample_rate))
    if data.size <= window:
        return 0
    hop = max(2048, window // 4)
    starts = list(range(0, data.size - window + 1, hop))
    if starts[-1] != data.size - window:
        starts.append(data.size - window)
    best_start = 0
    best_energy = -1.0
    for start in starts:
        block = data[start:start + window].astype(np.float64)
        energy = float(np.mean(block * block))
        if energy > best_energy:
            best_energy, best_start = energy, start
    return best_start


def _ltas_spectrum(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch-style RMS spectrum covering the complete supplied programme window."""
    data = _to_mono(audio).astype(np.float64)
    if data.size <= 1:
        return np.array([0.0], dtype=np.float32), np.array([EPS], dtype=np.float32)
    frame = min(16384, data.size)
    if frame < 1024:
        frame = data.size
    hop = max(1, frame // 2)
    window = np.hanning(frame).astype(np.float64)
    window_power = max(float(np.sum(window * window)), EPS)
    starts = list(range(0, max(1, data.size - frame + 1), hop))
    last = max(0, data.size - frame)
    if not starts or starts[-1] != last:
        starts.append(last)
    power_sum = None
    count = 0
    for start in starts:
        block = data[start:start + frame]
        if block.size < frame:
            block = np.pad(block, (0, frame - block.size))
        spectrum = np.fft.rfft(block * window)
        power = (np.abs(spectrum) ** 2) / window_power
        power_sum = power if power_sum is None else power_sum + power
        count += 1
    mean_power = power_sum / max(1, count)
    rms_spectrum = np.sqrt(np.maximum(mean_power, EPS)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1.0 / float(sample_rate)).astype(np.float32)
    return freqs, rms_spectrum


def _compensated_band_levels(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    freqs, spec = _ltas_spectrum(audio, sample_rate)
    compensation_db = COMPENSATION_DB_PER_OCTAVE * np.log2(np.maximum(freqs, 1.0) / 100.0)
    weighted = spec.astype(np.float64) * np.power(10.0, compensation_db / 20.0)
    levels: Dict[str, float] = {}
    for name, (low, high) in BAND_SPECS.items():
        mask = (freqs >= low) & (freqs < high)
        levels[name] = _amp_to_db(float(np.sqrt(np.mean(weighted[mask] ** 2)))) if np.any(mask) else -120.0
    finite_mid = [levels[name] for name in MIDLINE_BANDS if levels[name] > -119.0]
    reference = float(np.median(finite_mid)) if finite_mid else -120.0
    return {name: float(value - reference) for name, value in levels.items()}


def _raw_band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    freqs, spec = _ltas_spectrum(audio, sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    # Sum power, not FFT magnitude, so contributions add in an energy domain.
    return float(np.sum(spec[mask].astype(np.float64) ** 2))


def _segment(channel: LiveSharedMixChannel, start: int, end: int) -> np.ndarray:
    return _to_mono(channel.audio)[start:end].astype(np.float32)


def _audible_segment(channel: LiveSharedMixChannel, start: int, end: int) -> np.ndarray:
    if channel.muted:
        return np.zeros(max(0, end - start), dtype=np.float32)
    return (_segment(channel, start, end) * _db_to_amp(channel.fader_db)).astype(np.float32)


def _analysis_mix(channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> np.ndarray:
    n = max(0, end - start)
    mix = np.zeros(n, dtype=np.float32)
    for channel in channels:
        data = _audible_segment(channel, start, end)
        if data.size < n:
            data = np.pad(data, (0, n - data.size))
        mix += data[:n]
    return mix


def _band_shares(channels: Sequence[LiveSharedMixChannel], start: int, end: int, low_hz: float, high_hz: float) -> Dict[int, float]:
    energies: Dict[int, float] = {}
    for channel in channels:
        energies[channel.channel_id] = _raw_band_energy(_audible_segment(channel, start, end), channel.sample_rate, low_hz, high_hz)
    total = sum(energies.values())
    return {cid: (value / total if total > EPS else 0.0) for cid, value in energies.items()}


def _band_summary(channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> Dict[str, float]:
    if not channels:
        return {name: -120.0 for name in DISPLAY_CORRIDOR}
    levels = _compensated_band_levels(_analysis_mix(channels, start, end), channels[0].sample_rate)
    return {name: round(float(levels.get(name, -120.0)), 2) for name in DISPLAY_CORRIDOR}


def _lead_masking_state(channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> Dict[str, float]:
    lead_mid = other_mid = lead_bright = total_bright = 0.0
    for channel in channels:
        seg = _audible_segment(channel, start, end)
        mid = _raw_band_energy(seg, channel.sample_rate, 1500.0, 4000.0)
        bright = _raw_band_energy(seg, channel.sample_rate, 6000.0, 10000.0)
        total_bright += bright
        if channel.role == "lead_vocal":
            lead_mid += mid
            lead_bright += bright
        else:
            other_mid += mid
    return {
        "lead_share_1500_4000": float(lead_mid / (lead_mid + other_mid + EPS)),
        "accompaniment_share_1500_4000": float(other_mid / (lead_mid + other_mid + EPS)),
        "lead_sibilance_share_6000_10000": float(lead_bright / (total_bright + EPS)),
    }


def _top_culprits(channels: Sequence[LiveSharedMixChannel], start: int, end: int, low: float, high: float, *, allowed_roles: Optional[set[str]] = None, limit: int = 3):
    shares = _band_shares(channels, start, end, low, high)
    result = [(ch, shares.get(ch.channel_id, 0.0)) for ch in channels if allowed_roles is None or ch.role in allowed_roles]
    return sorted(result, key=lambda item: item[1], reverse=True)[:limit]


def _eq_band_for_freq(freq_hz: float) -> int:
    return 1 if freq_hz < 220.0 else 2 if freq_hz < 1200.0 else 3 if freq_hz < 5500.0 else 4


def _append_eq(actions: List[Any], decisions: List[Dict[str, Any]], channel: LiveSharedMixChannel, freq: float, delta: float, q: float, reason: str, config: LiveSharedMixConfig) -> None:
    if channel.muted or not channel.auto_corrections_enabled:
        return
    delta = float(np.clip(delta, -1.0, 1.0))
    if abs(delta) < config.min_action_db:
        return
    band = _eq_band_for_freq(freq)
    current = float(channel.current_eq_gain.get(band, 0.0))
    if not np.isfinite(current):
        current = 0.0
    target = round(float(np.clip(current + delta, -12.0, 12.0)), 2)
    channel.current_eq_gain[band] = target
    actions.append(ChannelEQMove(channel_id=channel.channel_id, band=band, freq_hz=float(freq), gain_db=target, q=float(q), reason=reason))
    decisions.append({"channel": channel.channel_id, "name": channel.name, "role": channel.role, "action": "eq", "reason": reason, "target_db": target})


def _append_fader(actions: List[Any], decisions: List[Dict[str, Any]], channel: LiveSharedMixChannel, delta: float, reason: str, config: LiveSharedMixConfig) -> None:
    if channel.muted or not channel.auto_corrections_enabled or not np.isfinite(channel.fader_db):
        return
    delta = float(np.clip(delta, -1.0, 1.0))
    target = float(np.clip(channel.fader_db + delta, -144.0, 0.0))
    if abs(target - channel.fader_db) < config.min_action_db:
        return
    channel.fader_db = target
    actions.append(ChannelFaderMove(channel_id=channel.channel_id, target_db=round(target, 2), is_lead=channel.role == "lead_vocal", reason=reason))
    decisions.append({"channel": channel.channel_id, "action": "fader", "reason": reason, "target_db": round(target, 2)})


def _append_hpf(actions: List[Any], decisions: List[Dict[str, Any]], channel: LiveSharedMixChannel, freq: float, reason: str) -> None:
    if channel.muted or not channel.auto_corrections_enabled or freq <= channel.current_hpf_hz + 5.0:
        return
    channel.current_hpf_hz = float(freq)
    actions.append(HighPassAdjust(channel_id=channel.channel_id, freq_hz=float(freq), enabled=True, reason=reason))
    decisions.append({"channel": channel.channel_id, "action": "hpf", "reason": reason, "target_hz": freq})


def _phase_log(label: str, channels: Sequence[LiveSharedMixChannel], start: int, end: int) -> Dict[str, Any]:
    return {
        "phase": label,
        "band_deviation_db": _band_summary(channels, start, end),
        "lead_masking": {k: round(v, 3) for k, v in _lead_masking_state(channels, start, end).items()},
    }


def _apply_low_end(channels, start, end, actions, decisions, config):
    summary = _band_summary(channels, start, end)
    if summary["50_100"] > DISPLAY_CORRIDOR["50_100"]["max"]:
        for channel, share in _top_culprits(channels, start, end, 50.0, 100.0, allowed_roles={"kick", "bass"}, limit=2):
            if share >= 0.25:
                _append_eq(actions, decisions, channel, 78.0 if channel.role == "kick" else 82.0, -1.0, 1.0, "50-100 Hz overweight: split kick/bass roles", config)
                _append_fader(actions, decisions, channel, -0.5, "50-100 Hz overweight: keep low-end anchor controlled", config)
                break
    # Even when the global compensated corridor is not exceeded, a synthetic or
    # sparse low-end-only programme needs a source-domain anchor correction.
    if set(ch.role for ch in channels) <= {"kick", "bass"} and len(channels) >= 2 and not any(isinstance(a, ChannelEQMove) for a in actions):
        culprit, share = _top_culprits(channels, start, end, 50.0, 110.0, allowed_roles={"kick", "bass"}, limit=1)[0]
        if share >= 0.45:
            _append_eq(actions, decisions, culprit, 80.0, -0.5, 1.0, "Low-end anchor: bounded source correction before master processing", config)


def _apply_lead_space(channels, start, end, actions, decisions, config):
    leads = [ch for ch in channels if ch.role == "lead_vocal"]
    if not leads:
        return
    mask = _lead_masking_state(channels, start, end)
    if mask["lead_share_1500_4000"] < 0.34:
        for channel, share in _top_culprits(channels, start, end, 1500.0, 4000.0, allowed_roles={"guitars", "playback", "bgv", "snare", "hi_hat", "ride", "overheads_room"}, limit=3):
            if share >= 0.10:
                _append_eq(actions, decisions, channel, 2500.0 if channel.role in {"guitars", "playback", "bgv"} else 3200.0, -1.0, 1.0, "Free 1.5-4 kHz space around lead instead of master EQ", config)
    if mask["lead_share_1500_4000"] < 0.28:
        for lead in leads:
            _append_fader(actions, decisions, lead, 0.5, "Small lead support after competitor EQ in 1.5-4 kHz", config)


def _apply_music_cleanup(channels, start, end, actions, decisions, config):
    summary = _band_summary(channels, start, end)
    if summary["200_500"] > DISPLAY_CORRIDOR["200_500"]["max"]:
        for channel, share in _top_culprits(channels, start, end, 200.0, 500.0, allowed_roles={"guitars", "playback", "bgv", "overheads_room", "lead_vocal"}, limit=3):
            if share >= 0.12:
                _append_eq(actions, decisions, channel, 320.0, -1.0, 0.9, "200-500 Hz blanket: clean culprit stem, not master EQ", config)
                if channel.role in {"guitars", "playback", "bgv", "overheads_room"}:
                    _append_hpf(actions, decisions, channel, 120.0 if channel.role in {"guitars", "playback"} else 150.0, "Secondary layer HPF from low-mid buildup")
                break


def _apply_air_cleanup(channels, start, end, actions, decisions, config):
    summary = _band_summary(channels, start, end)
    if summary["5000_8000"] <= DISPLAY_CORRIDOR["5000_8000"]["max"] and summary["8000_12000"] <= DISPLAY_CORRIDOR["8000_12000"]["max"]:
        return
    for channel, share in _top_culprits(channels, start, end, 6000.0, 10000.0, allowed_roles={"hi_hat", "ride", "overheads_room", "bgv", "playback", "lead_vocal"}, limit=2):
        if share >= 0.10:
            _append_eq(actions, decisions, channel, 7600.0, -0.8, 1.3, "Control dominant 6-10 kHz source before master processing", config)


def _mirror_eq(channels, start, end, actions, decisions, report, config):
    if not config.mirror_eq_enabled:
        report["mirror_eq"] = {"enabled": False, "reason": "disabled"}
        return
    eligible = [ch for ch in channels if ch.auto_corrections_enabled and not ch.muted]
    candidates: List[Tuple[LiveSharedMixChannel, LiveSharedMixChannel, float, float]] = []
    probe_bands = ((120.0, 250.0, 180.0), (700.0, 2000.0, 1200.0), (1800.0, 4500.0, 2500.0), (5000.0, 9000.0, 7000.0))
    for i, high_priority in enumerate(eligible):
        for low_priority in eligible[i + 1:]:
            a, b = (high_priority, low_priority) if high_priority.priority >= low_priority.priority else (low_priority, high_priority)
            if abs(a.priority - b.priority) < 0.05:
                # Stable tie-break: lead vocal wins against accompaniment.
                if b.role == "lead_vocal" and a.role != "lead_vocal":
                    a, b = b, a
                elif a.role != "lead_vocal" and b.role != "lead_vocal":
                    continue
            best = None
            for low, high, center in probe_bands:
                ea = _raw_band_energy(_audible_segment(a, start, end), a.sample_rate, low, high)
                eb = _raw_band_energy(_audible_segment(b, start, end), b.sample_rate, low, high)
                overlap = min(ea, eb)
                if best is None or overlap > best[0]:
                    best = (overlap, center)
            if best and best[0] > EPS:
                candidates.append((a, b, best[1], best[0]))
    candidates.sort(key=lambda item: item[3], reverse=True)
    planned = []
    limit = max(0, int(config.mirror_eq_max_actions_per_pass))
    for preferred, masker, freq, _ in candidates:
        if len(planned) + 2 > limit:
            break
        before = len(actions)
        _append_eq(actions, decisions, masker, freq, -1.0, 4.0, f"Mirror EQ cut masker: cross-adaptive overlap at {freq:.0f}Hz", config)
        _append_eq(actions, decisions, preferred, freq, 0.5, 2.0, f"Mirror EQ boost masked source: cross-adaptive overlap at {freq:.0f}Hz", config)
        if len(actions) > before:
            planned.extend([masker.channel_id, preferred.channel_id])
    report["mirror_eq"] = {
        "enabled": True,
        "mode": "cross_adaptive_full_mirror_eq",
        "candidate_count": len(candidates),
        "planned_candidates": planned,
        "principle": "cut lower-priority masker with narrower Q, boost higher-priority masked source more gently",
    }


def _append_master(actions: List[Any], report: Dict[str, Any], master_audio: Optional[np.ndarray], current: Optional[float], config: LiveSharedMixConfig) -> None:
    if master_audio is None or not config.correct_master_output:
        report["master"] = {"enabled": False, "reason": "no_master_reference_audio"}
        return
    peak = _peak_db(master_audio)
    report["master"] = {"enabled": True, "peak_dbfs": round(peak, 2), "rms_db": round(_rms_db(master_audio), 2), "peak_ceiling_dbfs": config.master_peak_ceiling_db, "principle": "master spectrum is a balance meter; source/stem fixes are preferred"}
    excess = peak - config.master_peak_ceiling_db
    if excess <= config.min_action_db:
        report["master"].update(action="none", reason="master peak inside live ceiling")
        return
    if current is None or not np.isfinite(current):
        report["master"].update(action="skip", reason="main fader readback unavailable")
        return
    cut = min(config.master_max_cut_db, excess)
    target = min(0.0, float(current) - cut)
    actions.append(MasterFaderMove(main_id=1, target_db=round(target, 2), reason="Master reference peak exceeds live ceiling; reduce Main 1 only"))
    report["master"].update(action="master_fader_cut", current_fader_db=round(float(current), 2), target_fader_db=round(target, 2), cut_db=round(cut, 2))


def _action_priority(action: Any) -> int:
    if isinstance(action, MasterFaderMove): return 0
    if isinstance(action, ChannelEQMove) and str(getattr(action, "reason", "")).startswith("Mirror EQ"): return 1
    if isinstance(action, ChannelFaderMove): return 2
    if isinstance(action, ChannelEQMove): return 3
    if isinstance(action, HighPassAdjust): return 4
    return 5


def _routing_audit(channels: Sequence[LiveSharedMixChannel], config: LiveSharedMixConfig) -> List[Dict[str, Any]]:
    result = []
    for channel in channels:
        raw = channel.raw_settings or {}
        generic = channel.name.strip().upper() in {"", f"CH{channel.channel_id}", f"CH {channel.channel_id}"}
        result.append({"channel": channel.channel_id, "name": channel.name, "role": channel.role, "input_routing": raw.get("input_routing") or {}, "main_send": raw.get("main_send") or {}, "name_generic": generic, "routing_write_enabled": bool(config.apply_routing_fixes), "rename_enabled": bool(config.rename_generic_channels)})
    return result


def build_live_shared_mix_plan(channels: Sequence[LiveSharedMixChannel], sample_rate: int, *, config: Optional[LiveSharedMixConfig] = None, master_audio: Optional[np.ndarray] = None, master_current_fader_db: Optional[float] = None) -> LiveSharedMixPlan:
    config = config or LiveSharedMixConfig()
    if not config.enabled:
        return LiveSharedMixPlan(report={"enabled": False, "reason": "disabled"})
    active = [ch for ch in channels if np.asarray(ch.audio).size and not ch.muted and np.isfinite(ch.fader_db) and ch.fader_db > -90.0 and _peak_db(ch.audio) > -65.0]
    if not active:
        return LiveSharedMixPlan(report={"enabled": True, "reason": "no_active_channels"})

    program_end = min(_to_mono(ch.audio).size for ch in active)
    full_mix = _analysis_mix(active, 0, program_end)
    start = _analysis_window_start(full_mix, sample_rate, config.analysis_window_sec)
    window = min(int(max(1.0, config.analysis_window_sec) * sample_rate), program_end)
    end = min(program_end, start + window)

    actions: List[Any] = []
    decisions: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "enabled": True,
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
            "all contribution estimates are audible/fader weighted",
            "small bounded fader/EQ/HPF moves only",
        ],
        "analysis_before": _phase_log("before", active, start, end),
        "phases": [],
        "decisions": decisions,
        "routing_audit": _routing_audit(active, config),
    }

    low = [ch for ch in active if ch.role in {"kick", "bass"}]
    if low:
        _apply_low_end(low, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("kick_bass_anchor", active, start, end))
    _apply_lead_space(active, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("lead_anchor", active, start, end))
    _apply_music_cleanup(active, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("music_layer", active, start, end))
    _apply_air_cleanup(active, start, end, actions, decisions, config)
    report["phases"].append(_phase_log("cymbal_air_layer", active, start, end))
    _mirror_eq(active, start, end, actions, decisions, report, config)
    report["analysis_after"] = _phase_log("predicted_after", active, start, end)
    _append_master(actions, report, master_audio, master_current_fader_db, config)

    ranked = sorted(enumerate(actions), key=lambda item: (_action_priority(item[1]), item[0]))
    limited = [action for _, action in ranked[:max(0, int(config.max_actions_per_pass))]]
    report["actions_requested"] = len(actions)
    report["actions_planned"] = len(limited)
    report["actions_truncated"] = max(0, len(actions) - len(limited))
    report["planned_action_types"] = [getattr(action, "action_type", type(action).__name__) for action in limited]
    return LiveSharedMixPlan(actions=limited, report=report)


def normalize_live_role(preset: str = "", source_role: str = "", name: str = "") -> str:
    text = " ".join((str(preset or ""), str(source_role or ""), str(name or ""))).lower()
    if "kick" in text: return "kick"
    if "snare" in text or " sn " in f" {text} ": return "snare"
    if "tom" in text: return "toms"
    if "hat" in text or "hihat" in text: return "hi_hat"
    if "ride" in text: return "ride"
    if any(token in text for token in ("overhead", "room", "ohl", "ohr")): return "overheads_room"
    if "bass" in text: return "bass"
    if "guitar" in text or "gtr" in text: return "guitars"
    if "lead" in text and ("vox" in text or "vocal" in text): return "lead_vocal"
    if any(token in text for token in ("vox", "vocal", "backs", "bgv")): return "bgv"
    if any(token in text for token in ("playback", "tracks", " pb ", "accordion", "keys", "synth")): return "playback"
    return "unknown"
