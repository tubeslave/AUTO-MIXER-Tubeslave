from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.signal import butter, lfilter, savgol_filter, sosfilt


MIN_MAG = 1e-12
SMART_EQ_METHOD_ID = "offline_smarteq_style_spectral_eq_v1"


@dataclass(frozen=True)
class EQBand:
    band_type: str
    frequency_hz: float
    gain_db: float = 0.0
    q: float = 0.707
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OfflineSmartEQConfig:
    fft_size: int = 8192
    hop_size: int = 4096
    band_slots: int = 24
    max_active_bands: int = 8
    min_move_db: float = 0.85
    max_cut_db: float = 4.0
    max_boost_db: float = 1.5
    correction_scale: float = 0.48
    boost_scale: float = 0.32
    min_frequency_hz: float = 35.0
    max_frequency_hz: float = 18000.0
    max_q: float = 3.2
    default_q: float = 1.15


@dataclass(frozen=True)
class OfflineSmartEQReport:
    method_id: str
    role: str
    sample_rate: int
    duration_sec: float
    band_slots: int
    active_bands: Tuple[EQBand, ...]
    highpass_hz: float | None
    smart_weights: Tuple[Dict[str, Any], ...]
    analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["active_bands"] = [band.to_dict() for band in self.active_bands]
        return data


def _amp_to_db(value: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(value), MIN_MAG))


def _db_to_amp(value: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(value) / 20.0)


def _mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1).astype(np.float32)


def _normalize_role(role: str) -> str:
    key = "".join(ch for ch in str(role).lower() if ch.isalnum())
    aliases = {
        "leadvocal": "vocal",
        "vocal": "vocal",
        "backingvocal": "backing_vocal",
        "backvox": "backing_vocal",
        "kick": "kick",
        "snare": "snare",
        "tom": "tom",
        "overheads": "overheads",
        "hihat": "overheads",
        "bass": "bass",
        "electricguitar": "guitar",
        "acousticguitar": "guitar",
        "guitar": "guitar",
        "accordion": "accordion",
        "keyboard": "keyboard",
        "playback": "playback",
    }
    return aliases.get(key, "general")


def _gaussian_log(freqs: np.ndarray, center_hz: float, width_oct: float, gain_db: float) -> np.ndarray:
    log_delta = np.log2(np.maximum(freqs, 1.0) / float(center_hz))
    return float(gain_db) * np.exp(-0.5 * np.square(log_delta / max(float(width_oct), 0.05)))


def _target_curve_db(freqs: np.ndarray, role: str) -> np.ndarray:
    role = _normalize_role(role)
    freqs = np.asarray(freqs, dtype=np.float64)
    curve = -0.65 * np.log2(np.maximum(freqs, 35.0) / 1000.0)
    curve = np.clip(curve, -4.0, 3.0)

    if role == "vocal":
        curve += _gaussian_log(freqs, 2800.0, 0.95, 1.3)
        curve += _gaussian_log(freqs, 250.0, 0.8, -1.8)
        curve += _gaussian_log(freqs, 7600.0, 0.75, -0.8)
    elif role == "backing_vocal":
        curve += _gaussian_log(freqs, 2400.0, 1.0, 0.7)
        curve += _gaussian_log(freqs, 260.0, 0.8, -1.6)
    elif role == "kick":
        curve += _gaussian_log(freqs, 65.0, 0.6, 2.0)
        curve += _gaussian_log(freqs, 320.0, 0.7, -1.8)
        curve += _gaussian_log(freqs, 3800.0, 0.8, 0.8)
    elif role == "snare":
        curve += _gaussian_log(freqs, 210.0, 0.7, 1.2)
        curve += _gaussian_log(freqs, 750.0, 0.8, -1.2)
        curve += _gaussian_log(freqs, 4300.0, 0.9, 1.1)
    elif role == "tom":
        curve += _gaussian_log(freqs, 130.0, 0.8, 1.4)
        curve += _gaussian_log(freqs, 450.0, 0.7, -1.5)
    elif role == "overheads":
        curve += _gaussian_log(freqs, 250.0, 1.0, -2.0)
        curve += _gaussian_log(freqs, 9000.0, 1.1, 0.9)
    elif role == "bass":
        curve += _gaussian_log(freqs, 85.0, 0.85, 1.9)
        curve += _gaussian_log(freqs, 280.0, 0.8, -0.8)
        curve += _gaussian_log(freqs, 3200.0, 1.0, -1.2)
    elif role == "guitar":
        curve += _gaussian_log(freqs, 180.0, 0.75, -1.3)
        curve += _gaussian_log(freqs, 2500.0, 1.0, 0.8)
        curve += _gaussian_log(freqs, 6500.0, 0.8, -0.9)
    elif role == "accordion":
        curve += _gaussian_log(freqs, 220.0, 0.8, -1.1)
        curve += _gaussian_log(freqs, 1800.0, 1.1, 0.8)
    elif role == "playback":
        curve *= 0.35
    elif role == "keyboard":
        curve += _gaussian_log(freqs, 180.0, 0.8, -0.8)
        curve += _gaussian_log(freqs, 4000.0, 1.0, 0.5)

    return curve.astype(np.float64)


def _role_highpass_hz(role: str) -> float | None:
    role = _normalize_role(role)
    return {
        "vocal": 85.0,
        "backing_vocal": 105.0,
        "guitar": 70.0,
        "accordion": 75.0,
        "keyboard": 45.0,
        "overheads": 125.0,
        "snare": 75.0,
        "tom": 45.0,
        "kick": 28.0,
        "bass": 32.0,
        "playback": 25.0,
    }.get(role)


def _smart_weights(role: str) -> Tuple[Dict[str, Any], ...]:
    role = _normalize_role(role)
    if role in {"vocal", "backing_vocal"}:
        return (
            {"frequency_hz": 2600.0, "width": 68, "strength": 62, "smoothing": 35, "adaptive": 0, "mode": "track"},
            {"frequency_hz": 260.0, "width": 42, "strength": -40, "smoothing": 45, "adaptive": 0, "mode": "track"},
        )
    if role in {"kick", "bass"}:
        return (
            {"frequency_hz": 85.0, "width": 72, "strength": 64, "smoothing": 40, "adaptive": 0, "mode": "track"},
            {"frequency_hz": 320.0, "width": 40, "strength": -45, "smoothing": 45, "adaptive": 0, "mode": "track"},
        )
    if role in {"snare", "tom"}:
        return (
            {"frequency_hz": 220.0, "width": 55, "strength": 45, "smoothing": 40, "adaptive": 0, "mode": "track"},
            {"frequency_hz": 4200.0, "width": 60, "strength": 40, "smoothing": 40, "adaptive": 0, "mode": "track"},
        )
    return (
        {"frequency_hz": 900.0, "width": 70, "strength": 45, "smoothing": 50, "adaptive": 0, "mode": "track"},
    )


def _ltas_db(mono: np.ndarray, sample_rate: int, config: OfflineSmartEQConfig) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(mono, dtype=np.float32)
    if x.size == 0:
        raise ValueError("audio is empty")
    fft_size = int(max(1024, config.fft_size))
    hop = int(max(256, config.hop_size))
    if x.size < fft_size:
        padded = np.zeros(fft_size, dtype=np.float32)
        padded[: x.size] = x
        x = padded

    window = np.hanning(fft_size).astype(np.float32)
    magnitudes: List[np.ndarray] = []
    for start in range(0, max(1, x.size - fft_size + 1), hop):
        block = x[start : start + fft_size]
        if block.size < fft_size:
            break
        if float(np.sqrt(np.mean(np.square(block))) + MIN_MAG) < 1e-7:
            continue
        spectrum = np.abs(np.fft.rfft(block * window)).astype(np.float64)
        magnitudes.append(spectrum)

    if not magnitudes:
        magnitudes = [np.abs(np.fft.rfft(np.zeros(fft_size, dtype=np.float32))).astype(np.float64)]

    mean_mag = np.mean(np.stack(magnitudes, axis=0), axis=0)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / float(sample_rate))
    mask = (freqs >= config.min_frequency_hz) & (freqs <= min(config.max_frequency_hz, sample_rate / 2.0))
    return freqs[mask], _amp_to_db(mean_mag[mask] / max(fft_size / 2.0, 1.0))


def _smooth_curve(values: np.ndarray) -> np.ndarray:
    if values.size < 7:
        return values
    window = min(31, values.size if values.size % 2 else values.size - 1)
    if window < 7:
        return values
    return savgol_filter(values, window_length=window, polyorder=2, mode="interp")


def _center_frequencies(config: OfflineSmartEQConfig, sample_rate: int) -> np.ndarray:
    hi = min(config.max_frequency_hz, sample_rate / 2.0 * 0.92)
    return np.geomspace(config.min_frequency_hz, hi, int(config.band_slots))


def _interpolate_log(freqs: np.ndarray, values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.interp(np.log2(centers), np.log2(freqs), values)


def design_offline_smarteq_eq(
    audio: np.ndarray,
    sample_rate: int,
    role: str,
    config: OfflineSmartEQConfig | None = None,
) -> OfflineSmartEQReport:
    config = config or OfflineSmartEQConfig()
    mono = _mono(audio)
    freqs, observed_db = _ltas_db(mono, sample_rate, config)

    norm_mask = (freqs >= 180.0) & (freqs <= min(6500.0, sample_rate / 2.0))
    observed_norm = observed_db - float(np.median(observed_db[norm_mask] if np.any(norm_mask) else observed_db))
    target_norm = _target_curve_db(freqs, role)
    residual = _smooth_curve(target_norm - observed_norm)

    centers = _center_frequencies(config, sample_rate)
    moves = _interpolate_log(freqs, residual, centers)
    active: List[EQBand] = []

    highpass = _role_highpass_hz(role)
    if highpass and highpass > 0:
        active.append(EQBand("highpass", highpass, 0.0, 0.707, f"{_normalize_role(role)} cleanup high-pass"))

    candidate_bands: List[EQBand] = []
    for center, move in zip(centers, moves):
        move = float(move)
        if abs(move) < config.min_move_db:
            continue
        scale = config.boost_scale if move > 0.0 else config.correction_scale
        gain = move * scale
        gain = float(np.clip(gain, -config.max_cut_db, config.max_boost_db))
        if abs(gain) < 0.35:
            continue
        q = float(np.clip(config.default_q + 0.12 * min(abs(gain), 4.0), 0.55, config.max_q))
        reason = "spectral target boost" if gain > 0 else "spectral imbalance cut"
        candidate_bands.append(EQBand("peaking", float(center), gain, q, reason))

    candidate_bands.sort(key=lambda band: abs(band.gain_db), reverse=True)
    remaining = max(0, int(config.max_active_bands) - len(active))
    active.extend(sorted(candidate_bands[:remaining], key=lambda band: band.frequency_hz))

    peak_freq = float(freqs[int(np.argmax(observed_db))]) if freqs.size else 0.0
    report = OfflineSmartEQReport(
        method_id=SMART_EQ_METHOD_ID,
        role=_normalize_role(role),
        sample_rate=int(sample_rate),
        duration_sec=round(float(mono.size) / float(sample_rate), 3),
        band_slots=int(config.band_slots),
        active_bands=tuple(active),
        highpass_hz=highpass,
        smart_weights=_smart_weights(role),
        analysis={
            "peak_frequency_hz": round(peak_freq, 3),
            "median_observed_db": round(float(np.median(observed_db)), 3),
            "max_requested_move_db": round(float(np.max(np.abs(moves))) if moves.size else 0.0, 3),
            "active_band_count": len(active),
        },
    )
    return report


def _peaking_coefficients(sample_rate: int, frequency_hz: float, gain_db: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    frequency_hz = float(np.clip(frequency_hz, 20.0, sample_rate / 2.0 * 0.95))
    q = float(np.clip(q, 0.05, 24.0))
    w0 = 2.0 * math.pi * frequency_hz / float(sample_rate)
    alpha = math.sin(w0) / (2.0 * q)
    amp = 10.0 ** (float(gain_db) / 40.0)
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * amp
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * amp
    a0 = 1.0 + alpha / amp
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / amp
    return (
        np.asarray([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64),
        np.asarray([1.0, a1 / a0, a2 / a0], dtype=np.float64),
    )


def _apply_band_mono(audio: np.ndarray, sample_rate: int, band: EQBand) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if band.band_type == "highpass":
        cutoff = float(np.clip(band.frequency_hz, 15.0, sample_rate / 2.0 * 0.45))
        sos = butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
        return sosfilt(sos, x).astype(np.float32)
    if band.band_type == "peaking" and abs(band.gain_db) > 1e-6:
        b, a = _peaking_coefficients(sample_rate, band.frequency_hz, band.gain_db, band.q)
        return lfilter(b, a, x).astype(np.float32)
    return x.astype(np.float32, copy=False)


def apply_offline_smarteq_eq(
    audio: np.ndarray,
    sample_rate: int,
    report_or_bands: OfflineSmartEQReport | Sequence[EQBand],
) -> np.ndarray:
    bands = report_or_bands.active_bands if isinstance(report_or_bands, OfflineSmartEQReport) else tuple(report_or_bands)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        out = arr
        for band in bands:
            out = _apply_band_mono(out, sample_rate, band)
        return out.astype(np.float32)
    channels = []
    for channel in range(arr.shape[1]):
        out_ch = arr[:, channel]
        for band in bands:
            out_ch = _apply_band_mono(out_ch, sample_rate, band)
        channels.append(out_ch)
    return np.stack(channels, axis=1).astype(np.float32)


def summarize_reports(reports: Mapping[str, OfflineSmartEQReport]) -> Dict[str, Any]:
    return {
        name: report.to_dict()
        for name, report in reports.items()
    }
