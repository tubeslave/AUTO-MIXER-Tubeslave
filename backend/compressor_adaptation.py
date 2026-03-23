"""
Compressor parameter adaptation from signal features.

Based on IMP (Intelligent Music Production) book:
- Threshold: Giannoulis et al. [53] - single-parameter control based on
  crest factor as primary signal feature.
- Attack/Release: Schneider & Hanson [21], Giannoulis et al. [53],
  Maddams et al. [48] - scaled based on crest factor AND spectral flux
  (percussive / high flux -> shorter, sustained / low flux -> longer).
- Ratio: Ma et al. [62] - function of percussivity / loudness range.
- Makeup gain: Giannoulis et al. [53] - EBU-loudness-based average
  control-voltage compensation.
- Knee: IMP 7.4.1 - configured based on estimated compression amount.
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
THR_SAFE_MAX = -6.0
ATTACK_MS_MIN, ATTACK_MS_MAX = 0, 250   # IMP 7.4.1: 5–250 ms typical
RELEASE_MS_MIN, RELEASE_MS_MAX = 5, 3000  # IMP 7.4.1: 5 ms – 3 s typical
KNEE_MIN, KNEE_MAX = 0, 5
GAIN_MIN, GAIN_MAX = -6, 12

# Max target gain reduction (IMP 7.4.1: "limiting peaks of up to 6 dB")
MAX_TARGET_GR_DB = 6.0
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
    Threshold adaptation based on crest factor (IMP [53]).

    Giannoulis et al. [53] showed that only threshold needs to be user-
    controlled when time constants are adapted automatically from the
    signal.  The crest factor (peak-to-RMS) is the primary signal feature
    that determines how far above the average level the compressor should
    begin acting.

    High crest (percussive) → higher threshold (let transients through).
    Low crest (sustained)   → lower threshold (steady compression).
    """
    crest = features.crest_factor_db
    rms = features.rms_db
    peak = max(features.true_peak_db, features.peak_db)

    if peak < -50:
        return max(THR_MIN, min(THR_SAFE_MAX, base_threshold_db))

    # Base threshold from RMS level and crest factor (IMP [53]):
    # Place threshold between RMS and peak, position controlled by crest.
    # High crest → closer to peak; low crest → closer to RMS.
    if crest > 20:
        # Very percussive: threshold just below peak
        thr = peak - 6.0
    elif crest > 12:
        # Percussive: threshold in upper third between RMS and peak
        thr = rms + crest * 0.66
    elif crest > 6:
        # Moderate: threshold in the middle
        thr = rms + crest * 0.5
    else:
        # Sustained / low crest: threshold just above RMS
        thr = rms + crest * 0.33

    return float(max(THR_MIN, min(THR_SAFE_MAX, thr)))


def adapt_attack(
    features: ChannelSignalFeatures,
    base_attack_ms: float,
) -> float:
    """
    Attack time based on crest factor and spectral flux (IMP [21, 48, 53]).

    "If a signal is highly transient or percussive, shorter time constants
    are preferred" (IMP 7.4.1).

    Schneider & Hanson [21] scaled attack based on crest factor.
    Giannoulis et al. [53] scaled based on modified crest factor or
    modified spectral flux, subsequently used by Maddams et al. [48].

    High spectral flux indicates rapid timbral changes and also
    suggests shorter time constants.
    """
    crest = features.crest_factor_db
    flux = getattr(features, 'spectral_flux', 0.0)

    # Percussive (high crest) -> short attack to catch transients.
    # Sustained (low crest)   -> long attack to preserve natural dynamics.
    if crest > 18:
        # Very percussive: fast attack
        attack = max(5.0, base_attack_ms * 0.3)
    elif crest > 12:
        # Percussive: moderately fast
        attack = max(5.0, base_attack_ms * 0.5)
    elif crest > 8:
        # Moderate: near base
        attack = base_attack_ms
    elif crest > 4:
        # Sustained: slower attack
        attack = base_attack_ms * 1.5
    else:
        # Very sustained: slow attack
        attack = base_attack_ms * 2.0

    # Spectral flux modifier (IMP [48, 53]):
    # High flux -> shorten attack further (rapid spectral changes).
    if flux > 0.5:
        attack *= 0.7
    elif flux > 0.3:
        attack *= 0.85

    return float(max(ATTACK_MS_MIN, min(ATTACK_MS_MAX, attack)))


def adapt_release(
    features: ChannelSignalFeatures,
    base_release_ms: float,
) -> float:
    """
    Release time based on crest factor, spectral flux and RMS (IMP [20, 21, 48, 53]).

    McNally [20] scaled release based on RMS values.
    Schneider & Hanson [21] scaled based on crest factor.
    Giannoulis et al. [53] scaled based on modified crest factor or
    modified spectral flux, subsequently used by Maddams et al. [48].

    Percussive -> shorter release (quick recovery after transient).
    Sustained  -> longer release (smooth, avoid pumping).
    High spectral flux -> shorter release.
    """
    crest = features.crest_factor_db
    flux = getattr(features, 'spectral_flux', 0.0)

    if crest > 18:
        # Very percussive: fast release
        release = max(RELEASE_MS_MIN, base_release_ms * 0.4)
    elif crest > 12:
        # Percussive: moderately fast
        release = max(RELEASE_MS_MIN, base_release_ms * 0.6)
    elif crest > 8:
        # Moderate: near base
        release = base_release_ms
    elif crest > 4:
        # Sustained: longer release to avoid pumping
        release = base_release_ms * 1.5
    else:
        # Very sustained: long release
        release = base_release_ms * 2.0

    # Spectral flux modifier (IMP [48, 53]):
    # High flux -> shorten release (rapid spectral changes need fast recovery).
    if flux > 0.5:
        release *= 0.7
    elif flux > 0.3:
        release *= 0.85

    return float(max(RELEASE_MS_MIN, min(RELEASE_MS_MAX, release)))


def adapt_ratio(
    features: ChannelSignalFeatures,
    base_ratio: float,
) -> float:
    """
    Ratio based on dynamic range / percussivity (IMP [48, 62]).

    Ma et al. [62]: "more compression is applied to percussive tracks"
    translates to "the ratio setting of the compressor is a particular
    function of a certain measure of percussivity."

    Wide dynamic range → higher ratio.
    Narrow dynamic range → lower ratio.
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

    return float(max(1.1, min(100.0, r)))


def adapt_makeup(
    threshold_db: float,
    ratio: float,
    lufs_integrated: float,
) -> float:
    """
    EBU-loudness-based makeup gain (IMP [53]).

    "EBU loudness-based make-up gain produced a better approximation of
    how professional mixing engineers would set the make-up gain" (IMP 7.4.1).

    Makeup = average gain reduction estimated from the difference between
    the integrated loudness and the threshold, divided by the ratio.
    """
    if ratio < 1.01:
        ratio = 2.0

    # Estimate average gain reduction from integrated loudness.
    # The amount above threshold that gets compressed:
    over = max(0.0, lufs_integrated - threshold_db)
    avg_gr = over * (1.0 - 1.0 / ratio)

    # Makeup compensates for the average GR (IMP [53]).
    makeup = avg_gr

    return float(max(GAIN_MIN, min(GAIN_MAX, MAX_MAKEUP_GAIN_DB, makeup)))


def adapt_knee(
    threshold_db: float,
    ratio: float,
    features: ChannelSignalFeatures,
    base_knee: int = 2,
) -> int:
    """
    Knee width based on estimated compression amount (IMP 7.4.1).

    "A soft knee enables a smoother transition between non-compressed and
    compressed parts of the signal.  The knee width should be configured
    based on the estimated amount of compression applied."

    Higher estimated compression → wider knee for transparency.
    """
    peak = max(features.true_peak_db, features.peak_db)
    if ratio < 1.01:
        return base_knee

    over = max(0.0, peak - threshold_db)
    estimated_gr = over * (1.0 - 1.0 / ratio)

    if estimated_gr > 8:
        knee = 4
    elif estimated_gr > 4:
        knee = 3
    elif estimated_gr > 2:
        knee = 2
    else:
        knee = 1

    return max(KNEE_MIN, min(KNEE_MAX, knee))


def adapt_params(
    features: ChannelSignalFeatures,
    base_preset: Dict[str, Any],
    target_gr_db: float = 6.0,
) -> Dict[str, Any]:
    """
    Full parameter adaptation based on IMP book methods.

    Returns dict with keys: threshold, ratio, ratio_wing, detector,
    attack_ms, release_ms, knee, gain.
    """
    ratio_base = base_preset.get("ratio", 3.0)
    attack_base = base_preset.get("attack_ms", 15)
    release_base = base_preset.get("release_ms", 150)
    base_knee = base_preset.get("knee", 2)

    # Ratio from dynamic range (IMP [48, 62])
    ratio = adapt_ratio(features, ratio_base)

    # Threshold from crest factor (IMP [53])
    threshold = adapt_threshold(features, base_preset.get("threshold", -15), ratio, target_gr_db)

    # Attack from crest factor (IMP [21, 53])
    attack_ms = adapt_attack(features, attack_base)

    # Release from crest factor (IMP [20, 21, 53])
    release_ms = adapt_release(features, release_base)

    # Knee from estimated compression amount (IMP 7.4.1)
    knee = adapt_knee(threshold, ratio, features, base_knee)

    # Makeup from EBU-loudness average GR (IMP [53])
    lufs = features.rms_db if features.rms_db > -70 else features.lufs_momentary
    gain = adapt_makeup(threshold, ratio, lufs)

    # Safety: cap expected GR (IMP 7.4.1: "limiting peaks of up to 6 dB")
    peak = max(features.true_peak_db, features.peak_db) if features.peak_db > -100 else lufs
    if peak > threshold and ratio > 1.01:
        over = peak - threshold
        expected_gr = over * (1.0 - 1.0 / ratio)
        if expected_gr > MAX_TARGET_GR_DB:
            over_allowed = MAX_TARGET_GR_DB / (1.0 - 1.0 / ratio)
            threshold_new = peak - over_allowed
            threshold = max(threshold, min(THR_SAFE_MAX, threshold_new))
            logger.info(
                f"adapt_params: GR capped ({expected_gr:.1f} dB > {MAX_TARGET_GR_DB} dB), "
                f"threshold raised to {threshold:.1f} dB"
            )
            gain = adapt_makeup(threshold, ratio, lufs)

    return {
        "threshold": threshold,
        "ratio": ratio,
        "ratio_wing": ratio_float_to_wing(ratio),
        "detector": str(base_preset.get("detector", "rms")).lower(),
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "knee": knee,
        "gain": gain,
    }
