"""
Neural mix feature extractor — extracts style features from reference mixes
for style transfer. Based on Martínez Ramírez et al. "Deep Learning for
Intelligent Audio Effects" and Steinmetz & Reiss "Style Transfer of Audio
Effects with Differentiable Signal Processing".
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MixStyle:
    """Extracted style features from a mix."""
    name: str
    # Spectral envelope (smoothed magnitude spectrum in dB)
    spectral_envelope: np.ndarray = field(default_factory=lambda: np.array([]))
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    # Dynamics
    loudness_lufs: float = -18.0
    loudness_range_lu: float = 8.0
    dynamic_range_db: float = 12.0
    crest_factor_db: float = 8.0
    # Spectral features
    spectral_centroid: float = 2000.0
    spectral_rolloff: float = 8000.0
    spectral_flatness: float = 0.3
    # Per-band characteristics
    band_levels: Dict[str, float] = field(default_factory=dict)
    band_dynamics: Dict[str, float] = field(default_factory=dict)
    # Stereo
    stereo_width: float = 0.5
    stereo_correlation: float = 0.8
    # Temporal
    avg_attack_ms: float = 10.0
    avg_release_ms: float = 100.0

class NeuralMixExtractor:
    """Extracts mix style features for style transfer."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self.window = np.hanning(fft_size)
        self.band_defs = {
            'sub': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
            'mid': (500, 2000), 'high_mid': (2000, 4000),
            'high': (4000, 8000), 'air': (8000, 20000)
        }

    def extract(self, audio: np.ndarray, name: str = 'reference') -> MixStyle:
        """Extract style features from audio."""
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0) if audio.shape[0] <= 2 else np.mean(audio, axis=1)
        else:
            mono = audio

        mono = mono.astype(np.float32)

        # Compute STFT
        n_frames = max(1, (len(mono) - self.fft_size) // self.hop_size + 1)
        magnitude_frames = []

        for i in range(n_frames):
            start = i * self.hop_size
            frame = mono[start:start + self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            windowed = frame * self.window
            spectrum = np.abs(np.fft.rfft(windowed))
            magnitude_frames.append(spectrum)

        magnitudes = np.array(magnitude_frames)
        avg_spectrum = np.mean(magnitudes, axis=0)

        # Spectral envelope (smoothed)
        spectral_envelope = self._smooth_spectrum(20 * np.log10(avg_spectrum + 1e-10))

        # Level and dynamics
        rms_per_frame = np.sqrt(np.mean(magnitudes ** 2, axis=1) + 1e-12)
        rms_db = 20 * np.log10(rms_per_frame + 1e-10)

        peak_db = float(20 * np.log10(np.max(np.abs(mono)) + 1e-10))
        loudness_est = float(np.mean(rms_db))

        valid_rms = rms_db[rms_db > -60]
        if len(valid_rms) > 3:
            dynamic_range = float(np.percentile(valid_rms, 95) - np.percentile(valid_rms, 5))
            loudness_range = float(np.percentile(valid_rms, 95) - np.percentile(valid_rms, 10))
        else:
            dynamic_range = 12.0
            loudness_range = 8.0

        crest = peak_db - loudness_est

        # Spectral features
        total_energy = np.sum(avg_spectrum)
        centroid = float(np.sum(self.freqs * avg_spectrum) / (total_energy + 1e-10))
        cumsum = np.cumsum(avg_spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = float(self.freqs[min(rolloff_idx, len(self.freqs) - 1)])

        # Spectral flatness
        log_spec = np.log(avg_spectrum + 1e-10)
        geometric_mean = np.exp(np.mean(log_spec))
        arithmetic_mean = np.mean(avg_spectrum)
        flatness = float(geometric_mean / (arithmetic_mean + 1e-10))

        # Per-band analysis
        band_levels = {}
        band_dynamics = {}
        for band_name, (lo, hi) in self.band_defs.items():
            mask = (self.freqs >= lo) & (self.freqs < hi)
            if np.any(mask):
                band_energy = np.sum(magnitudes[:, mask] ** 2, axis=1)
                band_db = 10 * np.log10(band_energy + 1e-12)
                band_levels[band_name] = float(np.mean(band_db))
                valid_band = band_db[band_db > -60]
                band_dynamics[band_name] = float(np.std(valid_band)) if len(valid_band) > 3 else 3.0
            else:
                band_levels[band_name] = -60.0
                band_dynamics[band_name] = 0.0

        # Stereo analysis
        stereo_width = 0.5
        stereo_corr = 0.8
        if audio.ndim == 2 and min(audio.shape) >= 2:
            if audio.shape[0] == 2:
                left, right = audio[0], audio[1]
            else:
                left, right = audio[:, 0], audio[:, 1]
            mid = (left + right) / 2
            side = (left - right) / 2
            mid_e = np.mean(mid ** 2)
            side_e = np.mean(side ** 2)
            stereo_width = float(np.sqrt(side_e / (mid_e + 1e-12)))
            corr = np.corrcoef(left[:min(len(left), 48000)], right[:min(len(right), 48000)])
            stereo_corr = float(corr[0, 1]) if corr.shape == (2, 2) else 0.8

        # Temporal dynamics estimation
        if len(rms_db) > 10:
            diffs = np.diff(rms_db)
            attacks = diffs[diffs > 0]
            releases = diffs[diffs < 0]
            frame_time_ms = (self.hop_size / self.sample_rate) * 1000
            avg_attack = float(frame_time_ms / (np.mean(attacks) + 1e-8)) if len(attacks) > 0 else 10.0
            avg_release = float(frame_time_ms / (abs(np.mean(releases)) + 1e-8)) if len(releases) > 0 else 100.0
        else:
            avg_attack = 10.0
            avg_release = 100.0

        return MixStyle(
            name=name,
            spectral_envelope=spectral_envelope,
            frequencies=self.freqs.copy(),
            loudness_lufs=loudness_est,
            loudness_range_lu=loudness_range,
            dynamic_range_db=dynamic_range,
            crest_factor_db=crest,
            spectral_centroid=centroid,
            spectral_rolloff=rolloff,
            spectral_flatness=flatness,
            band_levels=band_levels,
            band_dynamics=band_dynamics,
            stereo_width=stereo_width,
            stereo_correlation=stereo_corr,
            avg_attack_ms=max(0.1, min(200, avg_attack)),
            avg_release_ms=max(10, min(2000, avg_release)),
        )

    def _smooth_spectrum(self, spectrum_db: np.ndarray, octave_fraction: int = 3) -> np.ndarray:
        """Apply 1/N octave smoothing."""
        smoothed = np.copy(spectrum_db)
        for i in range(len(spectrum_db)):
            if self.freqs[i] < 20:
                continue
            ratio = 2 ** (1.0 / (2 * octave_fraction))
            lower = self.freqs[i] / ratio
            upper = self.freqs[i] * ratio
            mask = (self.freqs >= lower) & (self.freqs <= upper)
            if np.any(mask):
                smoothed[i] = np.mean(spectrum_db[mask])
        return smoothed

    def compute_distance(self, style_a: MixStyle, style_b: MixStyle) -> float:
        """Compute perceptual distance between two styles."""
        d_spec = 0.0
        if len(style_a.spectral_envelope) == len(style_b.spectral_envelope) and len(style_a.spectral_envelope) > 0:
            d_spec = np.sqrt(np.mean((style_a.spectral_envelope - style_b.spectral_envelope) ** 2))

        d_loud = abs(style_a.loudness_lufs - style_b.loudness_lufs)
        d_dyn = abs(style_a.dynamic_range_db - style_b.dynamic_range_db)
        d_cent = abs(style_a.spectral_centroid - style_b.spectral_centroid) / 1000
        d_width = abs(style_a.stereo_width - style_b.stereo_width) * 10

        return float(d_spec * 0.3 + d_loud * 0.25 + d_dyn * 0.2 + d_cent * 0.15 + d_width * 0.1)
