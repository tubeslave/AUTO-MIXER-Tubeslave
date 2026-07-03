"""AYAIC Mix-Monolith-inspired offline level-plane balancing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import amp_to_db, db_to_amp, ensure_stereo, limit_peak


DEFAULT_INPUT_OFFSETS_DB = {
    "kick": 3.0,
    "snare": 1.5,
    "bass": 3.0,
    "vocal": 6.0,
    "backing_vocal": 0.0,
    "guitars": -2.0,
    "keys": -4.0,
    "drums": -3.0,
    "unknown": -6.0,
}


def compute_input_gain_db(
    name: str,
    role: str,
    audio: np.ndarray,
    config: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return a safe learned level-plane gain for one stem."""

    cfg = dict(config or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    base_lufs = float(input_cfg.get("base_lufs", -24.0))
    max_boost = float(input_cfg.get("max_boost_db", 4.0))
    min_gain = float(input_cfg.get("min_gain_db", -24.0))
    peak_ceiling = float(input_cfg.get("per_channel_peak_ceiling_dbfs", -6.0))
    offsets = dict(DEFAULT_INPUT_OFFSETS_DB)
    offsets.update({str(key): float(value) for key, value in (input_cfg.get("offsets_db", {}) or {}).items()})
    key = _normalize_role(name, role)
    target = base_lufs + float(offsets.get(key, offsets["unknown"]))
    measured = _active_level_dbfs(audio)
    raw_gain = target - measured
    peak = float(np.max(np.abs(audio)) + 1e-12) if np.asarray(audio).size else 0.0
    peak_db = amp_to_db(peak)
    peak_limited_boost = peak_ceiling - peak_db
    gain = min(raw_gain, max_boost, peak_limited_boost)
    gain = max(min_gain, gain)
    report = {
        "method": "ayaic_level_plane_input",
        "channel": name,
        "role": role,
        "normalized_role": key,
        "target_level_dbfs": round(target, 3),
        "measured_active_dbfs": round(measured, 3),
        "peak_dbfs": round(peak_db, 3),
        "raw_gain_db": round(raw_gain, 3),
        "applied_gain_db": round(gain, 3),
    }
    return round(float(gain), 3), report


def apply_output_finish(
    audio: np.ndarray,
    sample_rate: int,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply bounded output-side cleanup: high-pass, mirror-EQ cuts, glue compression."""

    cfg = dict(config or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    if not bool(output_cfg.get("enabled", True)):
        return np.asarray(audio, dtype=np.float32), {"enabled": False, "actions": []}
    data = ensure_stereo(audio)
    actions: list[dict[str, Any]] = []
    high_pass_hz = float(output_cfg.get("high_pass_hz", 30.0))
    if high_pass_hz > 0:
        data = _high_pass(data, sample_rate, high_pass_hz)
        actions.append({"type": "high_pass", "freq_hz": high_pass_hz})
    if bool(output_cfg.get("mirror_eq_enabled", True)):
        profile = _spectral_profile(data, sample_rate)
        max_cut = abs(float(output_cfg.get("max_mirror_eq_cut_db", 1.5)))
        low_mid_excess = profile["low_mid_db"] - profile["mid_db"] - float(output_cfg.get("low_mid_margin_db", 1.5))
        if low_mid_excess > 0.25:
            gain = -min(max_cut, low_mid_excess * 0.45)
            data = _peaking_eq(data, sample_rate, 250.0, gain, 0.9)
            actions.append({"type": "mirror_eq", "band": "low_mid", "freq_hz": 250.0, "gain_db": round(gain, 3)})
        harsh_excess = profile["presence_db"] - profile["mid_db"] - float(output_cfg.get("presence_margin_db", 3.0))
        if harsh_excess > 0.25:
            gain = -min(max_cut, harsh_excess * 0.35)
            data = _peaking_eq(data, sample_rate, 3500.0, gain, 1.2)
            actions.append({"type": "mirror_eq", "band": "presence", "freq_hz": 3500.0, "gain_db": round(gain, 3)})
    if bool(output_cfg.get("glue_compressor_enabled", True)):
        data = _compressor(
            data,
            sample_rate,
            threshold_db=float(output_cfg.get("compressor_threshold_db", -18.0)),
            ratio=float(output_cfg.get("compressor_ratio", 1.35)),
            attack_ms=float(output_cfg.get("compressor_attack_ms", 25.0)),
            release_ms=float(output_cfg.get("compressor_release_ms", 180.0)),
            makeup_db=float(output_cfg.get("compressor_makeup_db", 0.4)),
        )
        actions.append(
            {
                "type": "glue_compressor",
                "threshold_db": float(output_cfg.get("compressor_threshold_db", -18.0)),
                "ratio": float(output_cfg.get("compressor_ratio", 1.35)),
            }
        )
    ceiling = float(output_cfg.get("peak_ceiling_dbfs", -1.0))
    data, trim = limit_peak(data, ceiling)
    if trim:
        actions.append({"type": "output_peak_trim", "gain_db": round(trim, 3), "ceiling_dbfs": ceiling})
    return data.astype(np.float32), {"enabled": True, "mode": "ayaic_output_finish", "actions": actions}


def _normalize_role(name: str, role: str) -> str:
    label = f"{name} {role}".lower().replace("_", " ").replace("-", " ")
    if "back" in label and ("vox" in label or "vocal" in label):
        return "backing_vocal"
    if "kick" in label:
        return "kick"
    if "snare" in label:
        return "snare"
    if "bass" in label:
        return "bass"
    if "vocal" in label or "vox" in label:
        return "vocal"
    if "guitar" in label or "gtr" in label:
        return "guitars"
    if "playback" in label or "accordion" in label or "keys" in label or "piano" in label:
        return "keys"
    if "tom" in label or "oh " in label or "overhead" in label or "drum" in label:
        return "drums"
    return "unknown"


def _active_level_dbfs(audio: np.ndarray) -> float:
    mono = np.mean(np.abs(ensure_stereo(audio)), axis=1)
    if mono.size == 0:
        return -120.0
    block = max(1024, min(8192, mono.size // 20 or 1024))
    usable = mono[: (mono.size // block) * block]
    if usable.size < block:
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        return amp_to_db(rms)
    frames = usable.reshape(-1, block)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    active = rms[rms > np.percentile(rms, 40)]
    if active.size == 0:
        active = rms
    return amp_to_db(float(np.percentile(active, 75)))


def _spectral_profile(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    mono = np.mean(ensure_stereo(audio), axis=1)
    if mono.size == 0:
        return {"low_mid_db": -120.0, "mid_db": -120.0, "presence_db": -120.0}
    window = np.hanning(min(mono.size, int(sample_rate * 8.0)))
    segment = mono[: window.size] * window
    spectrum = np.abs(np.fft.rfft(segment)) + 1e-12
    freqs = np.fft.rfftfreq(segment.size, 1.0 / float(sample_rate))

    def band_db(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return -120.0
        return amp_to_db(float(np.sqrt(np.mean(spectrum[mask] * spectrum[mask])) / max(1, segment.size)))

    return {
        "low_mid_db": band_db(180.0, 380.0),
        "mid_db": band_db(700.0, 1800.0),
        "presence_db": band_db(2500.0, 5200.0),
    }


def _high_pass(audio: np.ndarray, sample_rate: int, freq_hz: float) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt

        sos = butter(2, max(10.0, min(freq_hz, sample_rate * 0.45)), btype="highpass", fs=sample_rate, output="sos")
        return sosfilt(sos, ensure_stereo(audio), axis=0).astype(np.float32)
    except Exception:
        return ensure_stereo(audio)


def _peaking_eq(audio: np.ndarray, sample_rate: int, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
    data = ensure_stereo(audio)
    freq_hz = max(20.0, min(float(freq_hz), 0.45 * float(sample_rate)))
    a = 10.0 ** (float(gain_db) / 40.0)
    omega = 2.0 * np.pi * freq_hz / float(sample_rate)
    alpha = np.sin(omega) / (2.0 * max(0.1, q))
    cos_w = np.cos(omega)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    acoef = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    try:
        from scipy.signal import lfilter

        return lfilter(b, acoef, data, axis=0).astype(np.float32)
    except Exception:
        return data


def _compressor(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
) -> np.ndarray:
    data = ensure_stereo(audio)
    mono = np.mean(np.abs(data), axis=1)
    attack = np.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1000.0))
    release = np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))
    env = np.zeros_like(mono, dtype=np.float32)
    current = 0.0
    for idx, value in enumerate(mono):
        coeff = attack if value > current else release
        current = float(coeff * current + (1.0 - coeff) * value)
        env[idx] = current
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = np.maximum(0.0, env_db - float(threshold_db))
    reduction = over * (1.0 - 1.0 / max(1.0, float(ratio)))
    gain = np.power(10.0, (-reduction + float(makeup_db)) / 20.0)[:, None]
    return (data * gain).astype(np.float32)
