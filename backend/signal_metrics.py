"""
Comprehensive signal metrics framework for auto-mixing.

Provides per-channel multi-dimensional signal analysis:
- Level metrics: momentary/short-term/integrated LUFS, true peak, RMS, crest factor
- Dynamics: dynamic range, ADSR envelope, transient density/strength/regularity
- Spectral: centroid, rolloff, flatness, tilt, 7-band energy, flux, brightness
- Inter-channel: cross-correlation, coherence, spectral similarity, level difference

All LUFS measurements use K-weighting per ITU-R BS.1770-4.
True peak uses 4× oversampling per ITU-R BS.1770-4.
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)

try:
    from lufs_gain_staging import LUFSMeter, TruePeakMeter, KWeightingFilter
    HAS_LUFS_METERS = True
except ImportError:
    HAS_LUFS_METERS = False


# ── K-weighting filter (standalone, for when lufs_gain_staging unavailable) ──

def _k_weight(samples: np.ndarray, sample_rate: int = 48000) -> np.ndarray:
    """Apply K-weighting filter (ITU-R BS.1770-4) to samples."""
    if HAS_LUFS_METERS:
        kw = KWeightingFilter(sample_rate)
        return kw.process(samples)

    from scipy.signal import lfilter
    # Stage 1: Pre-filter (high shelf +4dB @ ~1681 Hz)
    if sample_rate == 48000:
        b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
        a1 = np.array([1.0, -1.69065929318241, 0.73248077421585])
    else:
        # Approximate for other sample rates
        b1 = np.array([1.0, 0.0, 0.0])
        a1 = np.array([1.0, 0.0, 0.0])
    y = lfilter(b1, a1, samples)

    # Stage 2: High-pass ~38 Hz
    if sample_rate == 48000:
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, -1.99004745483398, 0.99007225036621])
    else:
        b2 = np.array([1.0, -2.0, 1.0])
        a2 = np.array([1.0, -1.99, 0.99])
    return lfilter(b2, a2, y)


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class LevelMetrics:
    """Loudness and level measurements."""
    peak_db: float = -100.0
    true_peak_dbtp: float = -100.0
    rms_db: float = -100.0
    lufs_momentary: float = -100.0     # 400ms window
    lufs_short_term: float = -100.0    # 3s window
    lufs_integrated: float = -100.0    # Full duration with gating
    crest_factor_db: float = 0.0       # true_peak - rms
    loudness_range_lu: float = 0.0     # EBU R128 LRA


@dataclass
class DynamicsMetrics:
    """Dynamics and envelope analysis."""
    dynamic_range_db: float = 0.0
    attack_time_ms: float = 0.0
    decay_time_ms: float = 0.0
    sustain_level_db: float = -100.0
    release_time_ms: float = 0.0
    envelope_variance: float = 0.0
    transient_density: float = 0.0     # Transients per second
    transient_strength_db: float = 0.0
    transient_regularity: float = 0.0  # 0-1, rhythmic consistency
    peak_to_rms_ratio: float = 0.0


@dataclass
class SpectralMetrics:
    """Spectral analysis results."""
    centroid_hz: float = 0.0
    rolloff_hz: float = 0.0
    flatness: float = 0.0             # 0=tonal, 1=noise-like
    spectral_tilt_db: float = 0.0     # Slope of spectrum (positive=bright)
    brightness: float = 0.0           # Energy ratio above 4kHz
    warmth: float = 0.0               # Energy ratio 200-800Hz
    mud_ratio: float = 0.0            # Energy ratio 200-500Hz vs total
    presence_ratio: float = 0.0       # Energy ratio 2-5kHz vs total
    flux: float = 0.0                 # Frame-to-frame spectral change
    band_energy: Dict[str, float] = field(default_factory=dict)


@dataclass
class InterChannelMetrics:
    """Comparison metrics between two channels."""
    channel_a: int = 0
    channel_b: int = 0
    cross_correlation: float = 0.0     # -1..+1
    delay_samples: int = 0
    delay_ms: float = 0.0
    coherence: float = 0.0            # Magnitude-squared coherence (avg)
    spectral_similarity: float = 0.0  # Cosine similarity of spectra
    level_difference_db: float = 0.0  # LUFS difference
    phase_inverted: bool = False


@dataclass
class ChannelMetrics:
    """Complete metrics for a single channel."""
    channel: int
    timestamp: float = 0.0
    level: LevelMetrics = field(default_factory=LevelMetrics)
    dynamics: DynamicsMetrics = field(default_factory=DynamicsMetrics)
    spectral: SpectralMetrics = field(default_factory=SpectralMetrics)

    def to_dict(self) -> Dict:
        """Serialize all metrics to a flat dict for JSON/logging."""
        d = {"channel": self.channel, "timestamp": self.timestamp}
        for obj_name, obj in [("level", self.level), ("dynamics", self.dynamics),
                               ("spectral", self.spectral)]:
            for k, v in obj.__dict__.items():
                if isinstance(v, dict):
                    for bk, bv in v.items():
                        d[f"{obj_name}_{k}_{bk}"] = round(bv, 2) if isinstance(bv, float) else bv
                elif isinstance(v, float):
                    d[f"{obj_name}_{k}"] = round(v, 2)
                else:
                    d[f"{obj_name}_{k}"] = v
        return d


FREQ_BANDS = {
    'sub': (20, 60),
    'bass': (60, 250),
    'low_mid': (250, 500),
    'mid': (500, 2000),
    'high_mid': (2000, 4000),
    'presence': (4000, 8000),
    'air': (8000, 20000),
}


# ── Signal Analyzer ─────────────────────────────────────────────

class SignalAnalyzer:
    """Per-channel signal analyzer computing all metrics.

    Feed audio blocks via ``process()``; retrieve accumulated metrics
    via ``get_metrics()`` at any time.
    """

    def __init__(self, channel: int, sample_rate: int = 48000,
                 block_size: int = 1024):
        self.channel = channel
        self.sample_rate = sample_rate
        self.block_size = block_size

        # LUFS meters
        self._lufs_meter = LUFSMeter(sample_rate) if HAS_LUFS_METERS else None
        self._tp_meter = TruePeakMeter(sample_rate) if HAS_LUFS_METERS else None

        # Buffers for integrated/short-term LUFS
        self._momentary_window = int(sample_rate * 0.4)  # 400ms
        self._short_term_window = int(sample_rate * 3.0)  # 3s
        self._integrated_blocks: List[float] = []
        self._short_term_buffer = deque(maxlen=self._short_term_window)
        self._k_buffer = deque(maxlen=self._short_term_window)

        # Envelope / dynamics
        self._envelope_db = deque(maxlen=int(sample_rate * 5 / block_size))
        self._level_history = deque(maxlen=int(sample_rate * 10 / block_size))
        self._last_env_db = -100.0
        self._time_sec = 0.0

        # Transient tracking
        self._transient_times: List[float] = []
        self._transient_strengths: List[float] = []

        # ADSR state machine
        self._adsr_state = "idle"  # idle, attack, sustain, release
        self._adsr_attack_start = -100.0
        self._adsr_attack_start_time = 0.0
        self._adsr_peak_db = -100.0
        self._adsr_peak_time = 0.0
        self._adsr_sustain_levels: List[float] = []
        self._adsr_release_start_time = 0.0
        self._adsr_release_start_db = -100.0

        self._attack_ms = 0.0
        self._decay_ms = 0.0
        self._sustain_db = -100.0
        self._release_ms = 0.0

        # Spectral
        self._fft_size = max(2048, block_size)
        self._window = np.hanning(self._fft_size)
        self._freqs = np.fft.rfftfreq(self._fft_size, 1.0 / sample_rate)
        self._prev_spectrum: Optional[np.ndarray] = None

        # Accumulated peak metrics
        self._max_peak = -100.0
        self._max_true_peak = -100.0
        self._all_rms: List[float] = []

    def reset(self):
        """Reset all accumulated state."""
        if self._lufs_meter:
            self._lufs_meter.reset()
        if self._tp_meter:
            self._tp_meter.reset()
        self._integrated_blocks.clear()
        self._short_term_buffer.clear()
        self._k_buffer.clear()
        self._envelope_db.clear()
        self._level_history.clear()
        self._transient_times.clear()
        self._transient_strengths.clear()
        self._prev_spectrum = None
        self._time_sec = 0.0
        self._last_env_db = -100.0
        self._max_peak = -100.0
        self._max_true_peak = -100.0
        self._all_rms.clear()
        self._adsr_state = "idle"
        self._attack_ms = 0.0
        self._decay_ms = 0.0
        self._sustain_db = -100.0
        self._release_ms = 0.0

    def process(self, samples: np.ndarray):
        """Process a block of audio samples."""
        if len(samples) == 0:
            return
        samples = np.asarray(samples, dtype=np.float32)
        self._time_sec += len(samples) / self.sample_rate

        # ── Level metrics ────────────────────────────────────
        peak_linear = float(np.max(np.abs(samples)))
        peak_db = 20.0 * np.log10(peak_linear + 1e-10)
        self._max_peak = max(self._max_peak, peak_db)

        rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
        rms_db = 20.0 * np.log10(rms + 1e-10)
        self._all_rms.append(rms_db)

        # Momentary LUFS (400ms K-weighted)
        lufs_m = -100.0
        if self._lufs_meter:
            lufs_m = self._lufs_meter.process(samples)

        # True peak
        tp_db = -100.0
        if self._tp_meter:
            tp_db = self._tp_meter.process(samples)
            self._max_true_peak = max(self._max_true_peak, tp_db)

        # Store for integrated LUFS
        if lufs_m > -70.0:
            self._integrated_blocks.append(lufs_m)

        # K-weighted samples for short-term LUFS
        k_samples = _k_weight(samples, self.sample_rate)
        self._k_buffer.extend(k_samples.tolist())

        # ── Envelope / ADSR ──────────────────────────────────
        env_db = max(lufs_m, rms_db)
        if env_db > -70:
            self._envelope_db.append(env_db)
            self._level_history.append(env_db)

        self._update_adsr(env_db)
        self._detect_transients(env_db)
        self._last_env_db = env_db

        # ── Spectral ─────────────────────────────────────────
        self._analyze_spectrum(samples)

    def _update_adsr(self, env_db: float):
        """Simple ADSR state machine for envelope timing."""
        threshold = -40.0
        if self._adsr_state == "idle":
            if env_db > threshold:
                self._adsr_state = "attack"
                self._adsr_attack_start = env_db
                self._adsr_attack_start_time = self._time_sec
                self._adsr_peak_db = env_db
                self._adsr_peak_time = self._time_sec
        elif self._adsr_state == "attack":
            if env_db > self._adsr_peak_db:
                self._adsr_peak_db = env_db
                self._adsr_peak_time = self._time_sec
            elif env_db < self._adsr_peak_db - 3.0:
                # Peak passed, compute attack time
                self._attack_ms = (self._adsr_peak_time - self._adsr_attack_start_time) * 1000
                self._decay_ms = (self._time_sec - self._adsr_peak_time) * 1000
                self._adsr_state = "sustain"
                self._adsr_sustain_levels = [env_db]
        elif self._adsr_state == "sustain":
            if env_db > -70:
                self._adsr_sustain_levels.append(env_db)
                if len(self._adsr_sustain_levels) > 0:
                    self._sustain_db = float(np.mean(self._adsr_sustain_levels[-50:]))
            if env_db < threshold:
                self._adsr_state = "release"
                self._adsr_release_start_time = self._time_sec
                self._adsr_release_start_db = env_db
        elif self._adsr_state == "release":
            if env_db < -60 or env_db > self._adsr_release_start_db + 6:
                self._release_ms = (self._time_sec - self._adsr_release_start_time) * 1000
                self._adsr_state = "idle"

    def _detect_transients(self, env_db: float):
        """Detect transients from envelope rise."""
        if len(self._envelope_db) < 2:
            return
        rise = env_db - self._last_env_db
        if rise > 3.0:
            self._transient_times.append(self._time_sec)
            self._transient_strengths.append(min(rise, 30.0))
        cutoff = self._time_sec - 3.0
        while self._transient_times and self._transient_times[0] < cutoff:
            self._transient_times.pop(0)
            self._transient_strengths.pop(0)

    def _analyze_spectrum(self, samples: np.ndarray):
        """Compute spectral metrics."""
        if len(samples) < self._fft_size:
            padded = np.zeros(self._fft_size, dtype=np.float32)
            padded[:len(samples)] = samples
            samples = padded
        block = samples[-self._fft_size:] * self._window
        spectrum = np.abs(np.fft.rfft(block)) + 1e-10
        norm = np.sqrt(np.sum(spectrum ** 2))
        self._current_spectrum = spectrum
        self._current_norm_spectrum = spectrum / max(norm, 1e-10)

        # Flux
        if self._prev_spectrum is not None and len(self._prev_spectrum) == len(self._current_norm_spectrum):
            diff = self._current_norm_spectrum - self._prev_spectrum
            self._flux = float(np.sqrt(np.sum(diff ** 2)))
        else:
            self._flux = 0.0
        self._prev_spectrum = self._current_norm_spectrum.copy()

    def get_metrics(self) -> ChannelMetrics:
        """Compute and return all accumulated metrics."""
        metrics = ChannelMetrics(channel=self.channel, timestamp=time.time())

        # ── Level ────────────────────────────────────────────
        lm = metrics.level
        lm.peak_db = self._max_peak
        lm.true_peak_dbtp = self._max_true_peak
        if self._all_rms:
            lm.rms_db = float(np.mean(self._all_rms[-100:]))

        # Momentary (last value from meter)
        if self._lufs_meter:
            lm.lufs_momentary = self._lufs_meter._last_lufs if hasattr(self._lufs_meter, '_last_lufs') else -100.0
            if self._integrated_blocks:
                lm.lufs_momentary = self._integrated_blocks[-1]

        # Short-term (3s K-weighted)
        if len(self._k_buffer) >= self.sample_rate:
            k_arr = np.array(list(self._k_buffer), dtype=np.float32)
            ms = float(np.mean(k_arr ** 2) + 1e-12)
            lm.lufs_short_term = -0.691 + 10 * np.log10(ms)

        # Integrated (gated)
        lm.lufs_integrated = self._compute_integrated_lufs()

        # Crest factor
        if lm.rms_db > -90:
            lm.crest_factor_db = lm.true_peak_dbtp - lm.rms_db

        # Loudness Range (simplified)
        lm.loudness_range_lu = self._compute_loudness_range()

        # ── Dynamics ─────────────────────────────────────────
        dm = metrics.dynamics
        if len(self._level_history) > 10:
            valid = [x for x in self._level_history if x > -70]
            if valid:
                dm.dynamic_range_db = max(valid) - min(valid)

        dm.attack_time_ms = self._attack_ms
        dm.decay_time_ms = self._decay_ms
        dm.sustain_level_db = self._sustain_db
        dm.release_time_ms = self._release_ms

        if len(self._envelope_db) > 5:
            dm.envelope_variance = float(np.std(list(self._envelope_db)))

        elapsed = max(1.0, self._time_sec)
        dm.transient_density = len(self._transient_times) / min(3.0, elapsed)
        dm.transient_strength_db = float(np.mean(self._transient_strengths)) if self._transient_strengths else 0.0

        if len(self._transient_times) >= 3:
            intervals = np.diff(self._transient_times)
            if len(intervals) > 0 and np.std(intervals) > 0:
                dm.transient_regularity = 1.0 / (1.0 + float(np.std(intervals)))

        if lm.peak_db > -90 and lm.rms_db > -90:
            dm.peak_to_rms_ratio = lm.peak_db - lm.rms_db

        # ── Spectral ─────────────────────────────────────────
        sm = metrics.spectral
        if hasattr(self, '_current_spectrum'):
            spectrum = self._current_spectrum
            total_energy = float(np.sum(spectrum ** 2))

            if total_energy > 1e-10:
                sm.centroid_hz = float(np.sum(self._freqs * spectrum) / np.sum(spectrum))

                cumsum = np.cumsum(spectrum)
                idx85 = np.searchsorted(cumsum, 0.85 * cumsum[-1])
                sm.rolloff_hz = float(self._freqs[min(idx85, len(self._freqs) - 1)])

                geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
                arith_mean = np.mean(spectrum)
                sm.flatness = float(geo_mean / (arith_mean + 1e-10))

                # Spectral tilt (linear regression slope on log-magnitude)
                valid = self._freqs > 50
                if np.any(valid):
                    log_f = np.log10(self._freqs[valid] + 1)
                    log_m = 20 * np.log10(spectrum[valid])
                    if len(log_f) > 2:
                        slope = float(np.polyfit(log_f, log_m, 1)[0])
                        sm.spectral_tilt_db = slope

                # Band energies and ratios
                for name, (lo, hi) in FREQ_BANDS.items():
                    mask = (self._freqs >= lo) & (self._freqs < hi)
                    if np.any(mask):
                        e = float(np.sum(spectrum[mask] ** 2))
                        sm.band_energy[name] = 20.0 * np.log10(e + 1e-10)

                # Brightness: energy above 4kHz / total
                high_mask = self._freqs >= 4000
                sm.brightness = float(np.sum(spectrum[high_mask] ** 2) / total_energy)

                # Warmth: 200-800Hz / total
                warm_mask = (self._freqs >= 200) & (self._freqs < 800)
                sm.warmth = float(np.sum(spectrum[warm_mask] ** 2) / total_energy)

                # Mud ratio: 200-500Hz / total
                mud_mask = (self._freqs >= 200) & (self._freqs < 500)
                sm.mud_ratio = float(np.sum(spectrum[mud_mask] ** 2) / total_energy)

                # Presence ratio: 2-5kHz / total
                pres_mask = (self._freqs >= 2000) & (self._freqs < 5000)
                sm.presence_ratio = float(np.sum(spectrum[pres_mask] ** 2) / total_energy)

            sm.flux = getattr(self, '_flux', 0.0)

        return metrics

    def _compute_integrated_lufs(self) -> float:
        """Compute integrated LUFS with BS.1770-4 double gating."""
        if not self._integrated_blocks:
            return -100.0

        blocks = np.array(self._integrated_blocks)

        # Pass 1: absolute gate at -70 LUFS
        pass1 = blocks[blocks > -70.0]
        if len(pass1) == 0:
            return -100.0

        linear = 10.0 ** (pass1 / 10.0)
        ungated_mean = 10.0 * np.log10(np.mean(linear))

        # Pass 2: relative gate at ungated_mean - 10 LU
        relative_threshold = ungated_mean - 10.0
        pass2 = pass1[pass1 >= relative_threshold]
        if len(pass2) == 0:
            return float(ungated_mean)

        linear2 = 10.0 ** (pass2 / 10.0)
        return float(10.0 * np.log10(np.mean(linear2)))

    def _compute_loudness_range(self) -> float:
        """Compute simplified EBU R128 Loudness Range (LRA)."""
        if len(self._integrated_blocks) < 10:
            return 0.0

        blocks = np.array(self._integrated_blocks)
        # Gate at -70 LUFS
        gated = blocks[blocks > -70.0]
        if len(gated) < 10:
            return 0.0

        # Percentiles: 10th to 95th
        p10 = float(np.percentile(gated, 10))
        p95 = float(np.percentile(gated, 95))
        return max(0.0, p95 - p10)


# ── Inter-channel comparison ─────────────────────────────────────

def compare_channels(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    sample_rate: int = 48000,
    ch_a: int = 0,
    ch_b: int = 0,
) -> InterChannelMetrics:
    """Compute comparison metrics between two audio signals."""
    result = InterChannelMetrics(channel_a=ch_a, channel_b=ch_b)

    min_len = min(len(samples_a), len(samples_b))
    if min_len < 1024:
        return result

    a = samples_a[:min_len].astype(np.float32)
    b = samples_b[:min_len].astype(np.float32)

    # Cross-correlation (GCC-PHAT)
    n = len(a)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2

    X1 = np.fft.rfft(a, n=fft_size)
    X2 = np.fft.rfft(b, n=fft_size)

    cross = np.conj(X2) * X1
    magnitude = np.abs(cross) + 1e-10
    phat = cross / magnitude
    gcc = np.real(np.fft.irfft(phat, n=fft_size))

    max_delay = int(sample_rate * 0.020)
    center = fft_size // 2
    gcc_shifted = np.concatenate([gcc[fft_size - center:], gcc[:center + 1]])
    search_start = max(0, center - max_delay)
    search_end = min(len(gcc_shifted), center + max_delay + 1)
    search_region = gcc_shifted[search_start:search_end]

    peak_idx = int(np.argmax(np.abs(search_region)))
    peak_val = float(search_region[peak_idx])
    delay = peak_idx - (center - search_start)

    result.cross_correlation = peak_val
    result.delay_samples = delay
    result.delay_ms = abs(delay) / sample_rate * 1000
    result.phase_inverted = peak_val < -0.3

    # Magnitude-squared coherence (averaged)
    block = 2048
    n_blocks = max(1, min_len // block)
    coherence_sum = 0.0
    for i in range(n_blocks):
        s = i * block
        ba = a[s:s + block]
        bb = b[s:s + block]
        if len(ba) < block:
            break
        Fa = np.fft.rfft(ba)
        Fb = np.fft.rfft(bb)
        Pab = Fa * np.conj(Fb)
        Paa = np.abs(Fa) ** 2 + 1e-10
        Pbb = np.abs(Fb) ** 2 + 1e-10
        coh = np.abs(Pab) ** 2 / (Paa * Pbb)
        coherence_sum += float(np.mean(coh))
    result.coherence = coherence_sum / max(1, n_blocks)

    # Spectral similarity (cosine)
    Sa = np.abs(np.fft.rfft(a))
    Sb = np.abs(np.fft.rfft(b))
    dot = float(np.sum(Sa * Sb))
    norm_a = float(np.sqrt(np.sum(Sa ** 2)))
    norm_b = float(np.sqrt(np.sum(Sb ** 2)))
    if norm_a > 1e-10 and norm_b > 1e-10:
        result.spectral_similarity = dot / (norm_a * norm_b)

    # Level difference
    rms_a = float(np.sqrt(np.mean(a ** 2) + 1e-12))
    rms_b = float(np.sqrt(np.mean(b ** 2) + 1e-12))
    result.level_difference_db = 20.0 * np.log10(rms_a + 1e-10) - 20.0 * np.log10(rms_b + 1e-10)

    return result
