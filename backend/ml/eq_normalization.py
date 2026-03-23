"""
EQ normalization -- applies corrective EQ to match a target spectral profile.
Based on Ma et al. "Intelligent Multitrack Equalization" and Hafezi & Reiss
"Autonomous Multitrack Equalization Based on Masking Reduction".
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EQBand:
    """Single EQ band parameters."""
    frequency: float
    gain_db: float
    q: float
    band_type: str = 'peaking'  # 'peaking', 'low_shelf', 'high_shelf', 'low_pass', 'high_pass'


@dataclass
class EQProfile:
    """Complete EQ profile for a channel."""
    bands: List[EQBand]
    channel_id: int = 0
    label: str = ''


TARGET_CURVES: Dict[str, List[Tuple[float, float]]] = {
    'flat': [(20, 0), (100, 0), (1000, 0), (10000, 0), (20000, 0)],
    'warm': [(20, 2), (100, 1), (500, 0), (2000, -1), (8000, -2), (20000, -4)],
    'bright': [(20, -1), (100, 0), (1000, 0), (4000, 2), (10000, 3), (20000, 2)],
    'vocal_presence': [
        (20, -3), (100, -2), (300, -1), (800, 0),
        (2500, 3), (5000, 4), (10000, 2), (20000, 0),
    ],
    'hpf_300': [
        (20, -24), (100, -12), (300, 0), (1000, 0),
        (10000, 0), (20000, 0),
    ],
    'scoop_mid': [
        (20, 0), (200, 1), (500, -3), (1000, -2),
        (3000, 0), (10000, 1), (20000, 0),
    ],
    'kick_drum': [
        (20, 3), (60, 4), (100, 2), (250, -3), (400, -4),
        (3000, 2), (5000, 3), (10000, 0), (20000, -6),
    ],
    'acoustic_guitar': [
        (20, -12), (100, -3), (200, 0), (800, -2),
        (2000, 1), (5000, 2), (10000, 1), (20000, -2),
    ],
}


class EQNormalizer:
    """Computes corrective EQ to match a target spectral curve."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096,
                 n_bands: int = 6):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.n_bands = n_bands
        self.freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

    def analyze_spectrum(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < self.fft_size:
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        n_frames = len(audio) // self.fft_size
        if n_frames == 0:
            n_frames = 1
        avg_spectrum = np.zeros(self.fft_size // 2 + 1)
        window = np.hanning(self.fft_size)
        for i in range(n_frames):
            frame = audio[i * self.fft_size:(i + 1) * self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame * window))
            avg_spectrum += spectrum ** 2
        avg_spectrum = np.sqrt(avg_spectrum / n_frames)
        return 20 * np.log10(avg_spectrum + 1e-10)

    def interpolate_target(self, curve_name: str) -> np.ndarray:
        if curve_name not in TARGET_CURVES:
            curve_name = 'flat'
        points = TARGET_CURVES[curve_name]
        target_freqs = [p[0] for p in points]
        target_gains = [p[1] for p in points]
        return np.interp(self.freqs, target_freqs, target_gains)

    def compute_correction(self, audio: np.ndarray, target_curve: str = 'flat',
                           max_gain_db: float = 12.0) -> EQProfile:
        current_spectrum = self.analyze_spectrum(audio)
        target_spectrum = self.interpolate_target(target_curve)
        smoothed_current = self._smooth_spectrum(current_spectrum,
                                                  octave_fraction=3)
        diff = target_spectrum - smoothed_current
        diff_normalized = diff - np.mean(diff)
        diff_clamped = np.clip(diff_normalized, -max_gain_db, max_gain_db)
        bands = self._fit_eq_bands(diff_clamped)
        return EQProfile(bands=bands, label=f'correction_{target_curve}')

    def _smooth_spectrum(self, spectrum: np.ndarray,
                         octave_fraction: int = 3) -> np.ndarray:
        smoothed = np.copy(spectrum)
        for i in range(len(spectrum)):
            if self.freqs[i] < 20:
                continue
            lower = self.freqs[i] / (2 ** (1.0 / (2 * octave_fraction)))
            upper = self.freqs[i] * (2 ** (1.0 / (2 * octave_fraction)))
            mask = (self.freqs >= lower) & (self.freqs <= upper)
            if np.any(mask):
                smoothed[i] = np.mean(spectrum[mask])
        return smoothed

    def _fit_eq_bands(self, diff: np.ndarray) -> List[EQBand]:
        bands = []
        band_centers = np.geomspace(
            60, min(16000, self.sample_rate / 2 - 100), self.n_bands
        )
        for fc in band_centers:
            idx = np.argmin(np.abs(self.freqs - fc))
            q_width = fc / 2
            lower_idx = np.argmin(np.abs(self.freqs - (fc - q_width / 2)))
            upper_idx = np.argmin(np.abs(self.freqs - (fc + q_width / 2)))
            region = diff[lower_idx:upper_idx + 1]
            if len(region) == 0:
                gain = 0.0
            else:
                gain = float(np.mean(region))
            q = max(0.5, min(8.0, fc / (q_width + 1e-8)))
            if abs(gain) > 0.5:
                band_type = 'peaking'
                if fc < 100:
                    band_type = 'low_shelf'
                elif fc > 10000:
                    band_type = 'high_shelf'
                bands.append(EQBand(
                    frequency=float(fc), gain_db=round(gain, 1),
                    q=round(q, 2), band_type=band_type,
                ))
        return bands

    def apply_profile_to_osc(self, profile: EQProfile) -> List[Dict]:
        commands = []
        for i, band in enumerate(profile.bands):
            commands.append({
                'band': i + 1,
                'frequency': band.frequency,
                'gain_db': band.gain_db,
                'q': band.q,
                'type': band.band_type,
            })
        return commands
