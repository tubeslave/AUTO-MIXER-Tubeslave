"""
Mix quality assessment metrics.
Based on De Man & Reiss "A Semantic Approach to Autonomous Mixing" and
Maddams et al. "An Autonomous Method for Multi-track Dynamic Range Compression".
"""
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class MixQualityScore:
    """Overall quality score for a mix."""
    overall: float  # 0-100
    loudness: float
    dynamics: float
    spectral_balance: float
    stereo_width: float
    clarity: float
    headroom: float
    details: Dict[str, float]


class MixQualityAnalyzer:
    """Analyzes mix quality based on multiple perceptual metrics."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def analyze(self, mix_audio: np.ndarray,
                target_lufs: float = -18.0) -> MixQualityScore:
        if len(mix_audio) == 0:
            return MixQualityScore(0, 0, 0, 0, 0, 0, 0, {})
        loudness_score = self._score_loudness(mix_audio, target_lufs)
        dynamics_score = self._score_dynamics(mix_audio)
        spectral_score = self._score_spectral_balance(mix_audio)
        stereo_score = self._score_stereo(mix_audio)
        clarity_score = self._score_clarity(mix_audio)
        headroom_score = self._score_headroom(mix_audio)
        overall = (loudness_score * 0.2 + dynamics_score * 0.2 +
                   spectral_score * 0.2 + stereo_score * 0.1 +
                   clarity_score * 0.2 + headroom_score * 0.1)
        return MixQualityScore(
            overall=overall, loudness=loudness_score,
            dynamics=dynamics_score, spectral_balance=spectral_score,
            stereo_width=stereo_score, clarity=clarity_score,
            headroom=headroom_score,
            details={
                'loudness_score': loudness_score,
                'dynamics_score': dynamics_score,
                'spectral_balance_score': spectral_score,
                'stereo_width_score': stereo_score,
                'clarity_score': clarity_score,
                'headroom_score': headroom_score,
            }
        )

    def _score_loudness(self, audio: np.ndarray, target_lufs: float) -> float:
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        rms_db = 20 * np.log10(rms)
        deviation = abs(rms_db - target_lufs)
        return max(0, 100 - deviation * 5)

    def _score_dynamics(self, audio: np.ndarray) -> float:
        block_size = self.sample_rate // 10
        blocks = [audio[i:i + block_size]
                  for i in range(0, len(audio) - block_size, block_size)]
        if not blocks:
            return 50.0
        rms_vals = [20 * np.log10(np.sqrt(np.mean(b ** 2)) + 1e-12)
                    for b in blocks]
        rms_vals = [v for v in rms_vals if v > -60]
        if len(rms_vals) < 3:
            return 50.0
        dyn_range = max(rms_vals) - min(rms_vals)
        if 6 <= dyn_range <= 20:
            return 90.0
        elif dyn_range < 6:
            return max(30, 90 - (6 - dyn_range) * 10)
        else:
            return max(30, 90 - (dyn_range - 20) * 3)

    def _score_spectral_balance(self, audio: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        bands = {'low': (20, 250), 'mid': (250, 4000), 'high': (4000, 20000)}
        energies = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            energies[name] = np.sum(spectrum[mask] ** 2) if np.any(mask) else 1e-12
        total = sum(energies.values())
        ratios = {k: v / total for k, v in energies.items()}
        ideal = {'low': 0.45, 'mid': 0.40, 'high': 0.15}
        deviation = sum(abs(ratios[k] - ideal[k]) for k in ideal)
        return max(0, 100 - deviation * 200)

    def _score_stereo(self, audio: np.ndarray) -> float:
        if audio.ndim == 1:
            return 50.0
        if audio.ndim == 2 and audio.shape[0] == 2:
            left, right = audio[0], audio[1]
        elif audio.ndim == 2 and audio.shape[1] == 2:
            left, right = audio[:, 0], audio[:, 1]
        else:
            return 50.0
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_energy = np.mean(mid ** 2)
        side_energy = np.mean(side ** 2)
        ratio = side_energy / (mid_energy + 1e-12)
        if 0.1 <= ratio <= 0.5:
            return 90.0
        elif ratio < 0.05:
            return 40.0
        elif ratio > 0.8:
            return 50.0
        else:
            return 70.0

    def _score_clarity(self, audio: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        presence = (freqs >= 2000) & (freqs < 5000)
        mud = (freqs >= 200) & (freqs < 500)
        presence_energy = (np.sum(spectrum[presence] ** 2)
                           if np.any(presence) else 1e-12)
        mud_energy = np.sum(spectrum[mud] ** 2) if np.any(mud) else 1e-12
        ratio = presence_energy / (mud_energy + 1e-12)
        if ratio > 0.3:
            return min(95, 60 + ratio * 30)
        return max(30, ratio * 200)

    def _score_headroom(self, audio: np.ndarray) -> float:
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak + 1e-12)
        headroom = -peak_db
        if headroom >= 3:
            return 95.0
        elif headroom >= 1:
            return 80.0
        elif headroom >= 0.1:
            return 60.0
        else:
            return max(0, 40 - (peak_db * 10))
