"""Audio file and measurement helpers for offline pipeline code."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
from typing import Any

import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".aif", ".aiff", ".ogg"}


def safe_slug(value: str) -> str:
    """Return a filesystem-safe ASCII-ish slug."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._")
    return slug or "audio"


def audio_files(path: str | Path) -> list[Path]:
    """Return supported audio files in a directory."""

    root = Path(path).expanduser()
    if not root.exists():
        return []
    return sorted(
        item
        for item in root.iterdir()
        if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
    )


def amp_to_db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def db_to_amp(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def to_mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1)


def ensure_2d(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data[:, None]
    if data.ndim == 2:
        return data
    return data.reshape(-1, 1)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    data = ensure_2d(audio)
    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return data[:, :2].astype(np.float32, copy=False)


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    data = ensure_2d(audio)
    if len(data) == length:
        return data
    if len(data) > length:
        return data[:length]
    padding = np.zeros((length - len(data), data.shape[1]), dtype=np.float32)
    return np.vstack([data, padding])


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or source_rate <= 0 or target_rate <= 0 or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    data = ensure_2d(audio)
    duration = len(data) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=len(data), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    channels = [
        np.interp(target_x, source_x, data[:, channel]).astype(np.float32)
        for channel in range(data.shape[1])
    ]
    return np.stack(channels, axis=1)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with scipy when available, otherwise linear interpolation."""

    if source_rate == target_rate or source_rate <= 0 or target_rate <= 0:
        return np.asarray(audio, dtype=np.float32)
    try:
        from scipy.signal import resample_poly

        ratio = Fraction(int(target_rate), int(source_rate)).limit_denominator(1000)
        return resample_poly(
            ensure_2d(audio),
            ratio.numerator,
            ratio.denominator,
            axis=0,
        ).astype(np.float32)
    except Exception:
        return _resample_linear(audio, source_rate, target_rate)


def read_audio(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Read audio as float32, optionally resampling to target sample rate."""

    audio, sample_rate = sf.read(str(Path(path).expanduser()), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"{path} contains no audio")
    sample_rate = int(sample_rate)
    if target_sample_rate and sample_rate != int(target_sample_rate):
        audio = resample_audio(audio, sample_rate, int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return np.asarray(audio, dtype=np.float32), sample_rate


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    """Write a PCM24 WAV/AIFF/FLAC-compatible file, creating parents."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.asarray(audio, dtype=np.float32), int(sample_rate), subtype="PCM_24")
    return target


def limit_peak(audio: np.ndarray, ceiling_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    """Apply a transparent scalar trim if sample peak exceeds a ceiling."""

    data = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(data)) + 1e-12) if data.size else 0.0
    ceiling = db_to_amp(float(ceiling_dbfs))
    if peak <= ceiling or peak <= 0.0:
        return data, 0.0
    gain_db = amp_to_db(ceiling / peak)
    return (data * db_to_amp(gain_db)).astype(np.float32), gain_db


def measure_audio(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Return level/spectral/dynamics/stereo metrics using existing project code."""

    data = np.asarray(audio, dtype=np.float32)
    try:
        from mix_agent.analysis.dynamics import compute_dynamics_metrics
        from mix_agent.analysis.loudness import compute_level_metrics
        from mix_agent.analysis.spectral import compute_spectral_metrics
        from mix_agent.analysis.stereo import compute_stereo_metrics

        level, limitations = compute_level_metrics(data, int(sample_rate))
        spectral = compute_spectral_metrics(data, int(sample_rate))
        dynamics = compute_dynamics_metrics(data, int(sample_rate))
        stereo = compute_stereo_metrics(ensure_stereo(data), int(sample_rate))
        return {
            "level": level,
            "spectral": spectral,
            "dynamics": dynamics,
            "stereo": stereo,
            "limitations": limitations + list(stereo.get("limitations", [])),
        }
    except Exception as exc:
        peak = float(np.max(np.abs(data)) + 1e-12) if data.size else 0.0
        rms = float(np.sqrt(np.mean(data * data) + 1e-12)) if data.size else 0.0
        return {
            "level": {
                "peak_dbfs": amp_to_db(peak),
                "true_peak_dbtp": amp_to_db(peak),
                "rms_dbfs": amp_to_db(rms),
                "integrated_lufs": -0.691 + amp_to_db(rms),
                "headroom_db": -amp_to_db(peak),
                "clip_count": int(np.sum(np.abs(data) >= 0.999)),
            },
            "spectral": {},
            "dynamics": {},
            "stereo": {},
            "limitations": [f"fallback measurement used: {exc}"],
        }


def measure_audio_file(path: str | Path) -> dict[str, Any]:
    audio, sample_rate = read_audio(path)
    metrics = measure_audio(audio, sample_rate)
    metrics["sample_rate"] = sample_rate
    metrics["duration_sec"] = round(len(audio) / float(max(1, sample_rate)), 3)
    return metrics


def signal_quality_score(metrics: dict[str, Any]) -> float:
    """Map technical metrics to a conservative 0..1 quality proxy."""

    level = metrics.get("level", {})
    spectral = metrics.get("spectral", {})
    dynamics = metrics.get("dynamics", {})
    stereo = metrics.get("stereo", {})
    peak = float(level.get("true_peak_dbtp", level.get("peak_dbfs", -120.0)) or -120.0)
    headroom = float(level.get("headroom_db", 0.0) or 0.0)
    clip_count = int(level.get("clip_count", 0) or 0)
    mud = float(spectral.get("muddiness_proxy", 0.0) or 0.0)
    harsh = float(spectral.get("harshness_proxy", 0.0) or 0.0)
    pumping = float(dynamics.get("compression_pumping_proxy", 0.0) or 0.0)
    phase_risk = bool(stereo.get("phase_cancellation_risk", False))

    score = 1.0
    if clip_count:
        score -= 0.35
    score -= min(0.30, max(0.0, peak + 1.0) * 0.12)
    score -= min(0.18, max(0.0, 1.0 - headroom) * 0.08)
    score -= min(0.18, max(0.0, mud - 0.16) * 0.8)
    score -= min(0.18, max(0.0, harsh - 0.18) * 0.8)
    score -= min(0.12, max(0.0, pumping - 2.5) * 0.03)
    if phase_risk:
        score -= 0.15
    return float(max(0.0, min(1.0, score)))


def loudness_match_to(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float | None,
    peak_ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, float]:
    """Match integrated loudness to a target, then enforce a peak ceiling."""

    if target_lufs is None:
        return limit_peak(audio, peak_ceiling_dbfs)
    metrics = measure_audio(audio, sample_rate)
    current = metrics.get("level", {}).get("integrated_lufs")
    try:
        current_lufs = float(current)
    except (TypeError, ValueError):
        return limit_peak(audio, peak_ceiling_dbfs)
    if not np.isfinite(current_lufs):
        return limit_peak(audio, peak_ceiling_dbfs)
    gain_db = float(target_lufs) - current_lufs
    gain_db = max(-12.0, min(12.0, gain_db))
    matched = np.asarray(audio, dtype=np.float32) * db_to_amp(gain_db)
    limited, limit_gain_db = limit_peak(matched, peak_ceiling_dbfs)
    return limited, float(gain_db + limit_gain_db)
