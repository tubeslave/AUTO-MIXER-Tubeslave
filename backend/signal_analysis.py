"""
Signal analysis for Auto Compressor module.

Extracts level metrics, envelope, transients, and spectral features
from post-fader audio per channel. Uses LUFS/TruePeak from lufs_gain_staging.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

from lufs_gain_staging import LUFSMeter, TruePeakMeter

logger = logging.getLogger(__name__)

# Wing compressor ratio values (float) and their corresponding OSC string labels
WING_RATIO_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 20.0, float('inf')]
WING_RATIO_STRINGS = ["1.0:1", "1.5:1", "2.0:1", "3.0:1", "4.0:1", "6.0:1", "8.0:1", "10:1", "20:1", "inf:1"]


@dataclass
class ChannelSignalFeatures:
    """Extracted features from post-fader signal for one channel."""
    channel_id: int
    # Level metrics
    peak_db: float = -100.0
    true_peak_db: float = -100.0
    rms_db: float = -100.0
    lufs_momentary: float = -100.0
    crest_factor_db: float = 0.0
    dynamic_range_db: float = 0.0
    # Envelope
    attack_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    envelope_variance: float = 0.0
    # Transients
    transient_density: float = 0.0  # per second
    transient_strength: float = 0.0
    transient_regularity: float = 0.0  # 0-1, rhythm consistency
    # Spectral
    spectral_centroid_hz: float = 0.0
    spectral_rolloff_hz: float = 0.0
    spectral_flux: float = 0.0  # IMP [48, 53]: frame-to-frame spectral change
    band_energy: Dict[str, float] = field(default_factory=dict)
    # Running stats (for dynamic range)
    level_history: List[float] = field(default_factory=list)


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


class SpectralAnalyzerCompressor:
    """Lightweight spectral analysis for compressor (centroid, rolloff, band energy, flux).

    Spectral flux is a frame-to-frame measure of spectral change used by
    Giannoulis et al. [53] and Maddams et al. [48] (IMP 7.4) to adapt
    compressor attack/release times.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.window = np.hanning(fft_size)
        self.freq_bands = {
            'sub': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000),
            'air': (8000, 20000)
        }
        self._prev_spectrum: Optional[np.ndarray] = None

    def reset(self):
        self._prev_spectrum = None

    def analyze(self, samples: np.ndarray) -> Dict:
        if len(samples) < self.fft_size:
            samples = np.pad(samples.astype(np.float32), (0, self.fft_size - len(samples)))
        block = samples[-self.fft_size:] * self.window
        spectrum = np.abs(np.fft.rfft(block)) + 1e-10
        centroid = float(np.sum(self.freqs * spectrum) / np.sum(spectrum))
        cumsum = np.cumsum(spectrum)
        total = cumsum[-1]
        rolloff = 0.0
        if total > 1e-10:
            idx = np.searchsorted(cumsum, 0.85 * total)
            rolloff = float(self.freqs[min(idx, len(self.freqs) - 1)])
        band_energy = {}
        for name, (lo, hi) in self.freq_bands.items():
            mask = (self.freqs >= lo) & (self.freqs < hi)
            if np.any(mask):
                e = np.sum(spectrum[mask] ** 2)
                band_energy[name] = float(20 * np.log10(e + 1e-10))
            else:
                band_energy[name] = -100.0

        # Spectral flux: L2 norm of the difference between consecutive
        # normalised magnitude spectra (IMP [48, 53]).
        flux = 0.0
        norm = np.sqrt(np.sum(spectrum ** 2))
        norm_spectrum = spectrum / max(norm, 1e-10)
        if self._prev_spectrum is not None and len(self._prev_spectrum) == len(norm_spectrum):
            diff = norm_spectrum - self._prev_spectrum
            flux = float(np.sqrt(np.sum(diff ** 2)))
        self._prev_spectrum = norm_spectrum

        return {
            'centroid': centroid,
            'rolloff': rolloff,
            'band_energy': band_energy,
            'spectral_flux': flux,
        }


class SignalFeatureExtractor:
    """
    Extracts ChannelSignalFeatures from post-fader audio stream.
    One instance per channel (or reuse with reset between channels).
    """

    def __init__(self, channel_id: int, sample_rate: int = 48000, block_size: int = 1024):
        self.channel_id = channel_id
        self.sample_rate = sample_rate
        self.block_size = block_size
        # Meters
        self.lufs_meter = LUFSMeter(sample_rate)
        self.true_peak_meter = TruePeakMeter(sample_rate)
        self.spectral = SpectralAnalyzerCompressor(sample_rate, fft_size=2048)
        # Envelope: simple RMS per block in dB
        self.envelope_db: deque = deque(maxlen=max(512, sample_rate * 3 // block_size))  # ~3 sec
        self.level_history: deque = deque(maxlen=sample_rate * 2 // block_size)  # ~2 sec for dynamic range
        # Transient detection: threshold on derivative of envelope
        self.last_env_db = -100.0
        self.transient_times: List[float] = []
        self.transient_strengths: List[float] = []
        self._time_sec = 0.0

    def reset(self):
        self.lufs_meter.reset()
        self.true_peak_meter.reset()
        self.spectral.reset()
        self.envelope_db.clear()
        self.level_history.clear()
        self.transient_times.clear()
        self.transient_strengths.clear()
        self.last_env_db = -100.0
        self._time_sec = 0.0

    def process(self, samples: np.ndarray) -> ChannelSignalFeatures:
        if len(samples) == 0:
            return self._make_features()
        samples = np.asarray(samples, dtype=np.float32)
        lufs = self.lufs_meter.process(samples)
        true_peak_db = self.true_peak_meter.process(samples)
        rms_linear = np.sqrt(np.mean(samples ** 2) + 1e-12)
        rms_db = 20 * np.log10(rms_linear + 1e-10)
        peak_db = 20 * np.log10(np.max(np.abs(samples)) + 1e-10)

        self._time_sec += len(samples) / self.sample_rate
        env_db = max(lufs, rms_db)
        if env_db > -70:
            self.envelope_db.append(env_db)
            self.level_history.append(env_db)
        # Transient: sharp rise in envelope
        if len(self.envelope_db) >= 2:
            rise = env_db - self.last_env_db
            if rise > 3.0:  # dB rise in one block
                self.transient_times.append(self._time_sec)
                self.transient_strengths.append(min(rise, 20.0))
            self.last_env_db = env_db

        # Keep only recent transients for density (last 2 sec)
        cutoff = self._time_sec - 2.0
        while self.transient_times and self.transient_times[0] < cutoff:
            self.transient_times.pop(0)
            self.transient_strengths.pop(0)

        spectral = self.spectral.analyze(samples)

        # Build features
        crest = (peak_db - lufs) if lufs > -70 else 0.0
        dyn_range = 0.0
        if len(self.level_history) > 10:
            valid = [x for x in self.level_history if x > -70]
            if valid:
                dyn_range = max(valid) - min(valid)
        attack_ms = 0.0
        decay_ms = 0.0
        env_var = 0.0
        if len(self.envelope_db) > 20:
            arr = np.array(self.envelope_db)
            env_var = float(np.std(arr))
            # Crude attack: time from first above -30 to max
            above = np.where(arr > -30)[0]
            if len(above) > 0:
                first_above = int(above[0])
                max_idx = int(np.argmax(arr))
                if max_idx > first_above:
                    attack_ms = (max_idx - first_above) * len(samples) / self.sample_rate * 1000.0
                # Decay: from max to 50% down
                peak_val = arr[max_idx]
                tail = arr[max_idx:]
                below = np.where(tail < peak_val - 6)[0]
                if len(below) > 0:
                    decay_ms = int(below[0]) * len(samples) / self.sample_rate * 1000.0

        density = len(self.transient_times) / 2.0 if self._time_sec > 0.5 else 0.0  # per sec
        strength = float(np.mean(self.transient_strengths)) if self.transient_strengths else 0.0
        regularity = 0.0
        if len(self.transient_times) >= 3:
            intervals = np.diff(self.transient_times)
            if np.std(intervals) > 0:
                regularity = 1.0 / (1.0 + np.std(intervals))

        return ChannelSignalFeatures(
            channel_id=self.channel_id,
            peak_db=float(peak_db),
            true_peak_db=float(true_peak_db),
            rms_db=float(rms_db),
            lufs_momentary=float(lufs),
            crest_factor_db=float(crest),
            dynamic_range_db=float(dyn_range),
            attack_time_ms=attack_ms,
            decay_time_ms=decay_ms,
            envelope_variance=env_var,
            transient_density=density,
            transient_strength=strength,
            transient_regularity=regularity,
            spectral_centroid_hz=spectral['centroid'],
            spectral_rolloff_hz=spectral['rolloff'],
            spectral_flux=spectral.get('spectral_flux', 0.0),
            band_energy=spectral['band_energy'],
            level_history=list(self.level_history)[-100:]
        )

    def _make_features(self) -> ChannelSignalFeatures:
        return ChannelSignalFeatures(channel_id=self.channel_id)
