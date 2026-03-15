"""
Adaptive Dynamic Range Compression threshold from onset peak statistics.

Uses High Frequency Content (HFC) onset detection to analyze transient
behavior and set compression thresholds based on percentile statistics.
Instrument-specific percentiles and attack/release times ensure
appropriate dynamics processing for each source type.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Percentile thresholds per instrument category
# Higher percentile = threshold set higher = less compression
ONSET_PERCENTILES = {
    "kick": 85,
    "snare": 85,
    "hihat": 85,
    "toms": 85,
    "overheads": 80,
    "percussion": 82,
    "bass_guitar": 78,
    "electric_guitar": 75,
    "acoustic_guitar": 72,
    "keys": 72,
    "vocals": 70,
    "brass": 75,
    "strings": 68,
}

# Attack and release times (ms) per instrument type
# Tuned for musical dynamics preservation in live concert context
ATTACK_RELEASE_TIMES = {
    "kick": (0.5, 50.0),       # Very fast attack for transient punch
    "snare": (1.0, 60.0),      # Fast attack, moderate release
    "hihat": (0.5, 30.0),      # Very fast, short release
    "toms": (2.0, 80.0),       # Slightly slower attack, let body through
    "overheads": (10.0, 150.0), # Slow attack preserves transients
    "percussion": (1.0, 40.0),  # Fast attack for tight control
    "bass_guitar": (5.0, 120.0),    # Moderate attack, long release for sustain
    "electric_guitar": (8.0, 100.0), # Moderate, preserve pick attack
    "acoustic_guitar": (10.0, 120.0), # Slow attack, let strums breathe
    "keys": (8.0, 100.0),      # Moderate for dynamic expression
    "vocals": (5.0, 80.0),     # Moderate attack, smooth release
    "brass": (5.0, 90.0),      # Moderate, preserve brass attacks
    "strings": (15.0, 200.0),  # Very slow, preserve dynamics
}

DEFAULT_PERCENTILE = 75
DEFAULT_ATTACK_MS = 5.0
DEFAULT_RELEASE_MS = 100.0


def _hfc_onset_detection(audio, sr, hop_length=512):
    """
    High Frequency Content (HFC) onset detection function.

    HFC weights the spectrum by frequency, emphasizing transients
    which tend to have more high-frequency energy.

    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate
        hop_length: hop size for analysis frames

    Returns:
        onset_strength: 1D array of onset strength values per frame
        times: 1D array of corresponding time values in seconds
    """
    audio = np.asarray(audio, dtype=np.float64)
    n_fft = 2048
    window = np.hanning(n_fft)

    n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
    onset_strength = np.zeros(n_frames)

    prev_hfc = 0.0
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))

        # HFC: weight each bin by its frequency index (emphasizes high freq)
        freq_weights = np.arange(len(spectrum), dtype=np.float64)
        hfc = np.sum(spectrum * freq_weights)

        # Onset strength is the positive difference (flux) of HFC
        flux = max(0.0, hfc - prev_hfc)
        onset_strength[i] = flux
        prev_hfc = hfc

    times = np.arange(n_frames) * hop_length / sr
    return onset_strength, times


def _detect_onset_peaks(onset_strength, peak_threshold_ratio=0.3):
    """
    Find peaks in onset strength signal.

    Args:
        onset_strength: 1D array of onset strength values
        peak_threshold_ratio: minimum peak height as fraction of max

    Returns:
        peak_indices: array of frame indices where peaks occur
        peak_values: array of onset strength values at peaks
    """
    if len(onset_strength) < 3:
        return np.array([0]), np.array([onset_strength[0]] if len(onset_strength) > 0 else [0.0])

    threshold = peak_threshold_ratio * np.max(onset_strength)
    peaks = []
    values = []

    for i in range(1, len(onset_strength) - 1):
        if (
            onset_strength[i] > onset_strength[i - 1]
            and onset_strength[i] > onset_strength[i + 1]
            and onset_strength[i] > threshold
        ):
            peaks.append(i)
            values.append(onset_strength[i])

    if len(peaks) == 0:
        # Fall back to max value
        idx = np.argmax(onset_strength)
        return np.array([idx]), np.array([onset_strength[idx]])

    return np.array(peaks), np.array(values)


def compute_onset_threshold(audio, sr=48000, instrument_type=None):
    """
    Compute adaptive compression threshold from onset peak statistics.

    Analyzes the audio's transient peaks using HFC onset detection,
    then sets the threshold at a percentile of peak amplitudes. The
    percentile varies by instrument type (drums get higher percentile
    = less compression of transients).

    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate
        instrument_type: optional string for instrument-specific percentile

    Returns:
        threshold_db: compression threshold in dBFS
    """
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) == 0:
        return -20.0  # safe default

    # Normalize
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return -60.0  # silence

    # Get onset strength
    if HAS_LIBROSA:
        try:
            onset_env = librosa.onset.onset_strength(y=audio.astype(np.float32), sr=sr)
            hop_length = 512
            times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        except Exception:
            onset_env, times = _hfc_onset_detection(audio, sr)
    else:
        onset_env, times = _hfc_onset_detection(audio, sr)

    # Find peaks
    peak_indices, peak_values = _detect_onset_peaks(onset_env)

    if len(peak_values) == 0:
        return -20.0

    # Get the corresponding audio amplitudes at onset peaks
    hop_length = 512
    frame_size = 2048
    peak_amplitudes_db = []

    for idx in peak_indices:
        sample_pos = idx * hop_length
        start = max(0, sample_pos - frame_size // 2)
        end = min(len(audio), sample_pos + frame_size // 2)
        segment = audio[start:end]
        if len(segment) == 0:
            continue
        rms = np.sqrt(np.mean(segment ** 2) + 1e-10)
        db = 20.0 * np.log10(rms / peak + 1e-10)
        peak_amplitudes_db.append(db)

    if len(peak_amplitudes_db) == 0:
        return -20.0

    peak_amplitudes_db = np.array(peak_amplitudes_db)

    # Get instrument-specific percentile
    percentile = ONSET_PERCENTILES.get(instrument_type, DEFAULT_PERCENTILE)

    # Compute threshold at the specified percentile of peak amplitudes
    threshold_db = float(np.percentile(peak_amplitudes_db, percentile))

    # Clamp to reasonable range
    threshold_db = max(-60.0, min(-3.0, threshold_db))

    logger.debug(
        f"Onset threshold for {instrument_type}: {threshold_db:.1f} dB "
        f"(percentile={percentile}, n_peaks={len(peak_amplitudes_db)})"
    )

    return threshold_db


def get_attack_release(instrument_type=None):
    """
    Get recommended attack and release times for a given instrument type.

    Args:
        instrument_type: string instrument type

    Returns:
        (attack_ms, release_ms): tuple of float values in milliseconds
    """
    if instrument_type and instrument_type in ATTACK_RELEASE_TIMES:
        return ATTACK_RELEASE_TIMES[instrument_type]
    return (DEFAULT_ATTACK_MS, DEFAULT_RELEASE_MS)


def compute_drc_params(audio, sr=48000, instrument_type=None):
    """
    Compute full DRC parameter set from audio analysis.

    Combines onset-based threshold with instrument-specific attack/release
    to produce a complete set of compressor parameters.

    Args:
        audio: 1D numpy array
        sr: sample rate
        instrument_type: optional instrument type string

    Returns:
        dict with keys: threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_db
    """
    threshold_db = compute_onset_threshold(audio, sr, instrument_type)
    attack_ms, release_ms = get_attack_release(instrument_type)

    # Instrument-specific ratios
    ratio_map = {
        "kick": 4.0,
        "snare": 3.5,
        "hihat": 3.0,
        "toms": 3.5,
        "overheads": 2.0,
        "percussion": 3.0,
        "bass_guitar": 4.0,
        "electric_guitar": 3.0,
        "acoustic_guitar": 2.5,
        "keys": 2.5,
        "vocals": 3.0,
        "brass": 2.5,
        "strings": 2.0,
    }
    ratio = ratio_map.get(instrument_type, 3.0)

    # Knee width: wider for smoother sources
    knee_map = {
        "kick": 3.0,
        "snare": 3.0,
        "vocals": 6.0,
        "strings": 8.0,
        "acoustic_guitar": 6.0,
        "keys": 5.0,
    }
    knee_db = knee_map.get(instrument_type, 4.0)

    # Estimate makeup gain to compensate for gain reduction
    # Rough approximation: makeup = (threshold * (1 - 1/ratio)) / 2
    makeup_db = abs(threshold_db) * (1.0 - 1.0 / ratio) * 0.4
    makeup_db = min(makeup_db, 12.0)  # cap at 12 dB

    return {
        "threshold_db": threshold_db,
        "ratio": ratio,
        "attack_ms": attack_ms,
        "release_ms": release_ms,
        "knee_db": knee_db,
        "makeup_db": makeup_db,
    }
