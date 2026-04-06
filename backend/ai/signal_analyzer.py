"""Signal analysis utilities for the MixingAgent.

Analyzes real audio buffer data to derive EQ corrections, compression state,
and phase/delay offsets — all decisions based on measured signal, not presets.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Frequency band definitions (Hz) ──────────────────────────────────────────
BANDS: Dict[str, Tuple[float, float]] = {
    "sub":      (20.0,   80.0),
    "bass":     (80.0,  250.0),
    "low_mid":  (250.0, 800.0),
    "mid":      (800.0, 2500.0),
    "high_mid": (2500.0, 8000.0),
    "high":     (8000.0, 20000.0),
}

# Band → EQ mapping: (center_freq_hz, band_slot_1_4_or_shelf)
# band_slot: 1-4 = parametric band; "low_shelf" / "high_shelf"
BAND_EQ_MAP: Dict[str, Tuple[float, str]] = {
    "sub":      (60.0,    "low_shelf"),
    "bass":     (120.0,   "1"),
    "low_mid":  (300.0,   "2"),
    "mid":      (1200.0,  "3"),
    "high_mid": (4000.0,  "4"),
    "high":     (10000.0, "high_shelf"),
}

# Expected spectral shape per instrument (relative dB vs flat spectrum).
# Positive = this band should be louder than average; negative = quieter.
# These are targets derived from audio engineering knowledge.
INST_BAND_TARGETS: Dict[str, Dict[str, float]] = {
    "kick": {
        "sub": +4.0, "bass": +2.0, "low_mid": -3.0,
        "mid": -5.0, "high_mid": -2.0, "high": -6.0,
    },
    "snare": {
        "sub": -5.0, "bass": -2.0, "low_mid": +1.0,
        "mid": +1.0, "high_mid": +3.0, "high": +2.0,
    },
    "tom": {
        "sub": +2.0, "bass": +3.0, "low_mid": -1.0,
        "mid": -2.0, "high_mid": -3.0, "high": -6.0,
    },
    "hihat": {
        "sub": -8.0, "bass": -6.0, "low_mid": -2.0,
        "mid": +1.0, "high_mid": +4.0, "high": +5.0,
    },
    "ride": {
        "sub": -7.0, "bass": -4.0, "low_mid": -1.0,
        "mid": +2.0, "high_mid": +3.0, "high": +3.0,
    },
    "overheads": {
        "sub": -6.0, "bass": -3.0, "low_mid": -1.0,
        "mid": +1.0, "high_mid": +2.0, "high": +2.0,
    },
    "overhead": {
        "sub": -6.0, "bass": -3.0, "low_mid": -1.0,
        "mid": +1.0, "high_mid": +2.0, "high": +2.0,
    },
    "room": {
        "sub": -4.0, "bass": -2.0, "low_mid": +0.0,
        "mid": +0.0, "high_mid": -1.0, "high": -2.0,
    },
    "bass": {
        "sub": +4.0, "bass": +3.0, "low_mid": -1.0,
        "mid": -3.0, "high_mid": -5.0, "high": -8.0,
    },
    "bass_guitar": {
        "sub": +4.0, "bass": +3.0, "low_mid": -1.0,
        "mid": -3.0, "high_mid": -5.0, "high": -8.0,
    },
    "lead_vocal": {
        "sub": -8.0, "bass": -3.0, "low_mid": -2.0,
        "mid": +2.0, "high_mid": +2.0, "high": +1.0,
    },
    "leadvocal": {
        "sub": -8.0, "bass": -3.0, "low_mid": -2.0,
        "mid": +2.0, "high_mid": +2.0, "high": +1.0,
    },
    "back_vocal": {
        "sub": -8.0, "bass": -4.0, "low_mid": -2.0,
        "mid": +1.0, "high_mid": +1.0, "high": +0.5,
    },
    "backvocal": {
        "sub": -8.0, "bass": -4.0, "low_mid": -2.0,
        "mid": +1.0, "high_mid": +1.0, "high": +0.5,
    },
    "electricguitar": {
        "sub": -6.0, "bass": -2.0, "low_mid": -1.0,
        "mid": +2.0, "high_mid": +2.0, "high": -1.0,
    },
    "electric_guitar": {
        "sub": -6.0, "bass": -2.0, "low_mid": -1.0,
        "mid": +2.0, "high_mid": +2.0, "high": -1.0,
    },
    "accordion": {
        "sub": -6.0, "bass": -2.0, "low_mid": -1.0,
        "mid": +1.0, "high_mid": +1.0, "high": -1.0,
    },
    "playback": {
        "sub": +0.0, "bass": +0.0, "low_mid": +0.0,
        "mid": +0.0, "high_mid": +0.0, "high": +0.0,
    },
}


def compute_band_energies_from_spectrum(
    freqs: np.ndarray,
    mag_db: np.ndarray,
) -> Dict[str, float]:
    """Convert a spectrum (frequencies, magnitude_db) to per-band RMS-dB values.

    Args:
        freqs: frequency array from AudioCapture.get_spectrum()
        mag_db: magnitude_db array from AudioCapture.get_spectrum()

    Returns:
        dict of band_name → mean RMS dB in that band
    """
    result: Dict[str, float] = {}
    if freqs is None or mag_db is None or len(freqs) == 0:
        return {b: -100.0 for b in BANDS}

    for band, (f_low, f_high) in BANDS.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            result[band] = -100.0
            continue
        vals = mag_db[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            result[band] = -100.0
        else:
            result[band] = float(np.mean(vals))

    return result


def compute_eq_corrections(
    measured: Dict[str, float],
    inst_target: Dict[str, float],
    min_correction_db: float = 1.8,
    max_correction_db: float = 4.0,
    max_corrections: int = 3,
) -> List[Dict]:
    """Compute needed EQ corrections by comparing measured band balance to target.

    Only returns corrections where the deviation is significant enough.

    Args:
        measured: band → dB (from compute_band_energies_from_spectrum)
        inst_target: band → expected relative prominence (from INST_BAND_TARGETS)
        min_correction_db: minimum deviation to act on
        max_correction_db: cap correction magnitude
        max_corrections: maximum number of bands to correct per call

    Returns:
        list of dicts: {band, freq, gain, q, eq_type, band_slot}
        sorted by abs(gain) descending (most important first)
    """
    # Only use bands with actual signal
    valid = {b: v for b, v in measured.items() if v > -70.0}
    if len(valid) < 2:
        return []

    # Compute current spectral shape (deviation from mean)
    mean_level = float(np.mean(list(valid.values())))
    current_shape = {b: v - mean_level for b, v in valid.items()}

    corrections = []
    for band, target_rel in inst_target.items():
        if band not in current_shape:
            continue

        deviation = current_shape[band] - target_rel  # positive = too loud in this band
        if abs(deviation) < min_correction_db:
            continue

        needed_gain = -deviation  # correct: if too loud → cut
        needed_gain = max(-max_correction_db, min(max_correction_db, needed_gain))

        freq, slot = BAND_EQ_MAP.get(band, (1000.0, "3"))
        is_shelf = slot in ("low_shelf", "high_shelf")
        corrections.append({
            "band": band,
            "freq": freq,
            "gain": round(needed_gain, 1),
            "q": 0.9 if is_shelf else 1.5,
            "eq_type": slot if is_shelf else "band",
            "band_slot": slot,
        })

    # Sort by magnitude of correction (most needed first), limit
    corrections.sort(key=lambda x: -abs(x["gain"]))
    return corrections[:max_corrections]


def cross_correlate_for_delay(
    ref_buf: np.ndarray,
    other_buf: np.ndarray,
    sr: int,
    max_delay_ms: float = 25.0,
    min_correlation: float = 0.15,
) -> Tuple[float, float, bool]:
    """Find time delay between ref_buf (e.g. snare) and other_buf (e.g. kick/OH).

    Uses FFT-based cross-correlation. Returns:
        (delay_ms, correlation_strength, is_inverted)
        - delay_ms > 0: other_buf is delayed relative to ref (mic further away)
        - delay_ms < 0: other_buf leads ref (unlikely, ignore)
        - correlation_strength: 0..1 — how strong the bleed correlation is
        - is_inverted: True → other channel should have polarity inverted

    Only returns meaningful result if correlation_strength > min_correlation.
    Otherwise returns (0.0, 0.0, False).
    """
    n = min(len(ref_buf), len(other_buf))
    if n < 256:
        return 0.0, 0.0, False

    max_lag = max(1, int(sr * max_delay_ms / 1000.0))
    r = ref_buf[:n].astype(np.float32)
    o = other_buf[:n].astype(np.float32)

    # Normalize to unit variance
    r_std = float(np.std(r))
    o_std = float(np.std(o))
    if r_std < 1e-8 or o_std < 1e-8:
        return 0.0, 0.0, False

    r /= r_std
    o /= o_std

    # FFT cross-correlation (zero-pad to avoid circular wrap)
    # Zero-pad to 2n to avoid circular wrap-around aliasing
    pad = 2 * n
    R = np.fft.rfft(r, n=pad)
    O = np.fft.rfft(o, n=pad)

    # irfft(O * conj(R))[k] = sum_t o[t] * r[(t-k) mod pad]
    # This peaks at k = +delay when o is a DELAYED version of r (o lags r by delay).
    # No /N normalization needed — irfft already accounts for it internally such that
    # the result equals the sum (not the mean), so we normalize by energy below.
    corr = np.fft.irfft(O * np.conj(R))

    # Energy normalization: Cauchy-Schwarz max = sqrt(sum_r^2 * sum_o^2)
    energy_r = float(np.sum(r ** 2))
    energy_o = float(np.sum(o ** 2))
    norm_base = max(math.sqrt(energy_r * energy_o), 1e-12)

    # Positive lags 0..max_lag: other_buf is delayed relative to ref
    pos = corr[:max_lag + 1]
    peak_pos_idx = int(np.argmax(np.abs(pos)))
    peak_pos_val = float(pos[peak_pos_idx])

    # Negative lags: corr[pad - m] = lag -m (other_buf leads ref)
    # neg[0] → lag -max_lag, neg[-1] → lag -1
    neg = corr[pad - max_lag: pad]
    peak_neg_idx = int(np.argmax(np.abs(neg)))
    peak_neg_val = float(neg[peak_neg_idx])
    neg_lag = -(max_lag - peak_neg_idx)

    pos_strength = abs(peak_pos_val) / norm_base
    neg_strength = abs(peak_neg_val) / norm_base

    if neg_strength > pos_strength:
        lag = neg_lag
        strength = neg_strength
        # Inverted if the negative-lag peak is negative (anti-phase bleed)
        is_inverted = peak_neg_val < 0
    else:
        lag = peak_pos_idx
        strength = pos_strength
        is_inverted = peak_pos_val < 0

    strength = min(1.0, strength)
    if strength < min_correlation:
        return 0.0, 0.0, False

    delay_ms = float(lag) / sr * 1000.0
    return delay_ms, strength, is_inverted


def analyze_compression_state(
    peak_db: float,
    rms_db: float,
    crest_factor_db: float,
    current_threshold_db: float,
    inst_type: str = "unknown",
) -> Dict:
    """Analyze compression effectiveness from crest factor and level data.

    Crest factor = peak - RMS. Instrument-specific ideal ranges:
        - Drums (transient): 10-16 dB with good compression
        - Vocals/bass (steady): 6-12 dB with good compression
        - If crest factor too high → under-compressed (threshold too high)
        - If crest factor too low → over-compressed (threshold too low)

    Returns:
        dict: state ('ok'|'under'|'over'|'silence'), threshold_delta, confidence, crest_factor_db
    """
    if peak_db < -50.0:
        return {"state": "silence", "threshold_delta": 0.0, "confidence": 0.0,
                "crest_factor_db": crest_factor_db}

    is_transient = any(x in inst_type for x in ("kick", "snare", "tom", "drum"))

    # Ideal crest factor ranges by instrument class
    if is_transient:
        ideal_min, ideal_max = 10.0, 18.0
    else:
        ideal_min, ideal_max = 5.0, 12.0

    state = "ok"
    threshold_delta = 0.0
    confidence = 0.0

    if crest_factor_db > ideal_max + 4.0:
        state = "under"
        threshold_delta = -3.0  # lower threshold to catch more peaks
        confidence = 0.72
    elif crest_factor_db > ideal_max:
        state = "under"
        threshold_delta = -1.5
        confidence = 0.65
    elif crest_factor_db < ideal_min - 4.0:
        state = "over"
        threshold_delta = +3.0  # raise threshold
        confidence = 0.70
    elif crest_factor_db < ideal_min:
        state = "over"
        threshold_delta = +1.5
        confidence = 0.62

    return {
        "state": state,
        "threshold_delta": threshold_delta,
        "confidence": confidence,
        "crest_factor_db": crest_factor_db,
        "ideal_range": (ideal_min, ideal_max),
    }


def smooth_level_adjustment(
    current_lufs: float,
    target_lufs: float,
    prev_adjustment: float = 0.0,
    alpha: float = 0.4,
    max_step_db: float = 0.5,
    deadband_db: float = 1.2,
) -> float:
    """Compute smooth fader adjustment toward LUFS target.

    Uses exponential smoothing with a hard step limiter.
    Returns the dB adjustment to apply this cycle (e.g. -0.5, +0.3).
    Returns 0.0 if within deadband.

    Args:
        current_lufs: measured LUFS momentary
        target_lufs: target LUFS for this instrument
        prev_adjustment: last adjustment applied (for smoothing)
        alpha: smoothing factor (0=slow, 1=instant)
        max_step_db: max dB adjustment per cycle
        deadband_db: ignore deviations smaller than this
    """
    diff = current_lufs - target_lufs  # positive = too loud
    if abs(diff) < deadband_db:
        return 0.0

    # Raw adjustment needed
    raw = -diff * alpha  # negative of error (reduce if too loud)

    # Apply smoothing with previous
    smoothed = (1 - alpha) * prev_adjustment + alpha * raw

    # Hard step limit
    clamped = max(-max_step_db, min(max_step_db, smoothed))
    return round(clamped, 2)
