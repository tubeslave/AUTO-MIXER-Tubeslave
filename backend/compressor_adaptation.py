"""
Compressor parameter adaptation from signal features and context.

Formulas: threshold from peak/RMS/target GR; attack/release from envelope and BPM;
ratio from dynamic range and genre; makeup from GR.
"""

import logging
from typing import Dict, Any, Optional

from signal_analysis import ChannelSignalFeatures

logger = logging.getLogger(__name__)

# Wing ratio: float -> nearest OSC string
WING_RATIO_STRINGS = ["1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"]
WING_RATIO_VALUES = [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0]

# Limits for Wing
THR_MIN, THR_MAX = -60, 0
# Safe max threshold: never set above -6 dB to avoid high GR on hot signals
THR_SAFE_MAX = -6.0
ATTACK_MS_MIN, ATTACK_MS_MAX = 0, 120
RELEASE_MS_MIN, RELEASE_MS_MAX = 4, 4000
KNEE_MIN, KNEE_MAX = 0, 5
GAIN_MIN, GAIN_MAX = -6, 12

# Max target gain reduction to avoid visible over-compression on compressor meters
MAX_TARGET_GR_DB = 6.0
# Max makeup gain to avoid excessive gain on some channels
MAX_MAKEUP_GAIN_DB = 6.0


def ratio_float_to_wing(ratio: float) -> str:
    """Convert float ratio to nearest Wing OSC ratio string."""
    if ratio <= 0:
        return "1.1"
    best_idx = 0
    best_diff = abs(WING_RATIO_VALUES[0] - ratio)
    for i, v in enumerate(WING_RATIO_VALUES):
        d = abs(v - ratio)
        if d < best_diff:
            best_diff = d
            best_idx = i
    return WING_RATIO_STRINGS[best_idx]


def adapt_threshold(
    features: ChannelSignalFeatures,
    base_threshold_db: float,
    ratio: float,
    target_gr_db: float = 6.0,
) -> float:
    """
    Threshold = Peak - (Ratio * Target_GR) / (Ratio - 1).
    Corrections: high crest -> raise threshold; low crest / unstable envelope -> lower.
    """
    peak = max(features.true_peak_db, features.peak_db)
    if peak < -50:
        return max(THR_MIN, min(THR_MAX, base_threshold_db))
    if ratio <= 1:
        ratio = 2.0
    thr = peak - (ratio * target_gr_db) / (ratio - 1)
    if features.crest_factor_db > 15:
        thr += 3.0
    elif features.crest_factor_db > 12:
        thr += 2.0
    elif features.crest_factor_db < 6:
        thr -= 2.5
    elif features.crest_factor_db < 8:
        thr -= 1.5
    if features.envelope_variance > 5.0:  # unstable dynamics
        thr -= 2.0
    # Cap threshold so we never set very high threshold (e.g. -5 dB) which causes high GR
    return float(max(THR_MIN, min(THR_SAFE_MAX, thr)))


def adapt_attack(
    features: ChannelSignalFeatures,
    base_attack_ms: float,
    task: str,
) -> float:
    """
    Preserve transients: attack > signal attack + 5 ms.
    Control transients: attack < signal attack.
    Punch (drums): 20–40 ms.
    """
    sig_attack = features.attack_time_ms
    if task == "punch":
        attack = max(20.0, min(40.0, base_attack_ms))
    elif task == "control":
        attack = min(base_attack_ms, sig_attack * 0.8) if sig_attack > 0 else base_attack_ms
    else:  # gentle / base / broadcast
        attack = max(base_attack_ms, sig_attack + 5.0) if sig_attack > 0 else base_attack_ms
    return float(max(ATTACK_MS_MIN, min(ATTACK_MS_MAX, attack)))


def adapt_release(
    features: ChannelSignalFeatures,
    base_release_ms: float,
    bpm: Optional[float] = None,
) -> float:
    """
    Release_base = 60000 / BPM / 4 (1/16 note) if BPM set.
    Release = clip(signal_decay * 1.5, Release_base * 0.5, Release_base * 2).
    Without BPM use signal decay and base.
    """
    decay = features.decay_time_ms
    if bpm and bpm > 0:
        release_base = 60000.0 / bpm / 4.0
        release = max(release_base * 0.5, min(release_base * 2.0, (decay * 1.5) if decay > 0 else release_base))
    else:
        release = (decay * 1.5) if decay > 10 else base_release_ms
    return float(max(RELEASE_MS_MIN, min(RELEASE_MS_MAX, release)))


def adapt_ratio(
    features: ChannelSignalFeatures,
    base_ratio: float,
    genre_factor: float = 1.0,
    mix_density_factor: float = 1.0,
) -> float:
    """
    Wide dynamic range -> higher ratio; narrow -> lower.
    Apply genre and mix density factors.
    """
    dyn = features.dynamic_range_db
    if dyn > 30:
        r = base_ratio * 1.2
    elif dyn > 20:
        r = base_ratio * 1.05
    elif dyn < 10:
        r = base_ratio * 0.85
    else:
        r = base_ratio
    r = r * genre_factor * mix_density_factor
    return float(max(1.1, min(100.0, r)))


def adapt_makeup(
    threshold_db: float,
    ratio: float,
    rms_db: float,
    peak_db: Optional[float] = None,
    compensation: float = 0.85,
) -> float:
    """
    Calculate makeup gain based on expected gain reduction.
    
    Uses both RMS and peak to estimate gain reduction more accurately:
    - RMS-based GR: for steady-state compression
    - Peak-based GR: for transient compression
    - Final makeup = weighted average * compensation factor
    
    Args:
        threshold_db: Compressor threshold
        ratio: Compressor ratio
        rms_db: RMS level of signal
        peak_db: Peak level of signal (optional, uses RMS if not provided)
        compensation: Compensation factor (0.8-0.9 typical, 0.85 default)
    
    Returns:
        Makeup gain in dB
    """
    if ratio < 1.01:
        ratio = 2.0
    
    # Calculate gain reduction based on RMS (steady-state)
    over_rms = max(0.0, rms_db - threshold_db)
    gr_rms = over_rms * (1.0 - 1.0 / ratio)
    
    # Calculate gain reduction based on peak (transients)
    if peak_db is not None and peak_db > threshold_db:
        over_peak = max(0.0, peak_db - threshold_db)
        gr_peak = over_peak * (1.0 - 1.0 / ratio)
        # Weighted average: 60% RMS (steady), 40% peak (transients)
        gr_avg = gr_rms * 0.6 + gr_peak * 0.4
    else:
        gr_avg = gr_rms
    
    # Apply compensation factor (typically 0.8-0.9 to avoid over-compensation)
    makeup = gr_avg * compensation
    
    # Ensure minimum makeup gain if compression is active
    if gr_avg > 1.0:  # If we have significant compression
        makeup = max(makeup, gr_avg * 0.5)  # At least 50% of GR
    
    return float(max(GAIN_MIN, min(GAIN_MAX, MAX_MAKEUP_GAIN_DB, makeup)))


def adapt_params(
    features: ChannelSignalFeatures,
    base_preset: Dict[str, Any],
    task: str = "base",
    target_gr_db: float = 6.0,
    bpm: Optional[float] = None,
    genre_factor: float = 1.0,
    mix_density_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Full adaptation: threshold, ratio, attack, release, knee, makeup.
    Returns dict with keys: threshold, ratio (float), ratio_wing (str), attack_ms, release_ms, knee, gain.
    """
    thr_base = base_preset.get("threshold", -15)
    ratio_base = base_preset.get("ratio", 3.0)
    attack_base = base_preset.get("attack_ms", 15)
    release_base = base_preset.get("release_ms", 150)
    knee = base_preset.get("knee", 2)
    knee = max(KNEE_MIN, min(KNEE_MAX, knee))

    # Cap target GR so compressor meters don't show excessive GR
    target_gr_capped = min(float(target_gr_db), MAX_TARGET_GR_DB)

    ratio = adapt_ratio(features, ratio_base, genre_factor, mix_density_factor)
    threshold = adapt_threshold(features, thr_base, ratio, target_gr_capped)
    attack_ms = adapt_attack(features, attack_base, task)
    release_ms = adapt_release(features, release_base, bpm)
    rms = features.rms_db if features.rms_db > -70 else features.lufs_momentary
    peak = max(features.true_peak_db, features.peak_db) if features.peak_db > -100 else rms
    gain = adapt_makeup(threshold, ratio, rms, peak_db=peak)

    # Post-step: if expected GR would exceed max, raise threshold to cap GR
    if peak > threshold and ratio > 1.01:
        over = peak - threshold
        expected_gr = over * (1.0 - 1.0 / ratio)
        if expected_gr > MAX_TARGET_GR_DB:
            # Raise threshold so that expected_gr <= MAX_TARGET_GR_DB
            over_allowed = MAX_TARGET_GR_DB / (1.0 - 1.0 / ratio)
            threshold_new = peak - over_allowed
            threshold = max(threshold, min(THR_SAFE_MAX, threshold_new))
            logger.info(f"adapt_params: high GR capped (expected {expected_gr:.1f}dB > {MAX_TARGET_GR_DB}dB), threshold raised to {threshold:.1f}dB")
            gain = adapt_makeup(threshold, ratio, rms, peak_db=peak)

    return {
        "threshold": threshold,
        "ratio": ratio,
        "ratio_wing": ratio_float_to_wing(ratio),
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "knee": knee,
        "gain": gain,
    }
