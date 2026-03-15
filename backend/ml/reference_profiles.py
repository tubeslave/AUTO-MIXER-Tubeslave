"""
Reference mix profiles -- stores spectral and loudness characteristics of reference mixes
for A/B comparison and style transfer targets.
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ReferenceProfile:
    """Spectral and loudness profile from a reference mix."""
    name: str
    genre: str
    spectral_envelope: np.ndarray  # dB values at frequency bins
    frequencies: np.ndarray
    lufs_integrated: float
    lufs_range: float
    true_peak_db: float
    dynamic_range_db: float
    crest_factor_db: float
    band_energies: Dict[str, float] = field(default_factory=dict)
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    stereo_width: float = 0.0
    description: str = ''


BUILTIN_PROFILES: Dict[str, Dict] = {
    'modern_rock': {
        'genre': 'rock',
        'lufs_integrated': -14.0,
        'lufs_range': 6.0,
        'true_peak_db': -0.5,
        'dynamic_range_db': 8.0,
        'crest_factor_db': 6.0,
        'spectral_centroid': 2200.0,
        'spectral_rolloff': 8500.0,
        'stereo_width': 0.6,
        'band_energies': {
            'sub': -20, 'bass': -8, 'low_mid': -6, 'mid': -4,
            'high_mid': -8, 'high': -14, 'air': -22,
        },
        'description': 'Loud, punchy modern rock with compressed dynamics',
    },
    'jazz_live': {
        'genre': 'jazz',
        'lufs_integrated': -20.0,
        'lufs_range': 12.0,
        'true_peak_db': -2.0,
        'dynamic_range_db': 18.0,
        'crest_factor_db': 12.0,
        'spectral_centroid': 1800.0,
        'spectral_rolloff': 7000.0,
        'stereo_width': 0.7,
        'band_energies': {
            'sub': -18, 'bass': -6, 'low_mid': -4, 'mid': -3,
            'high_mid': -6, 'high': -12, 'air': -20,
        },
        'description': 'Dynamic jazz with wide stereo image and natural sound',
    },
    'pop_broadcast': {
        'genre': 'pop',
        'lufs_integrated': -16.0,
        'lufs_range': 5.0,
        'true_peak_db': -1.0,
        'dynamic_range_db': 6.0,
        'crest_factor_db': 5.0,
        'spectral_centroid': 2500.0,
        'spectral_rolloff': 10000.0,
        'stereo_width': 0.5,
        'band_energies': {
            'sub': -22, 'bass': -10, 'low_mid': -6, 'mid': -3,
            'high_mid': -5, 'high': -10, 'air': -18,
        },
        'description': 'Polished pop mix for broadcast/streaming',
    },
    'worship': {
        'genre': 'worship',
        'lufs_integrated': -18.0,
        'lufs_range': 8.0,
        'true_peak_db': -1.0,
        'dynamic_range_db': 10.0,
        'crest_factor_db': 8.0,
        'spectral_centroid': 2000.0,
        'spectral_rolloff': 9000.0,
        'stereo_width': 0.55,
        'band_energies': {
            'sub': -20, 'bass': -8, 'low_mid': -5, 'mid': -3,
            'high_mid': -6, 'high': -12, 'air': -20,
        },
        'description': 'Warm worship mix with vocal clarity and controlled dynamics',
    },
    'edm_festival': {
        'genre': 'edm',
        'lufs_integrated': -12.0,
        'lufs_range': 4.0,
        'true_peak_db': -0.3,
        'dynamic_range_db': 5.0,
        'crest_factor_db': 4.0,
        'spectral_centroid': 2800.0,
        'spectral_rolloff': 12000.0,
        'stereo_width': 0.65,
        'band_energies': {
            'sub': -6, 'bass': -4, 'low_mid': -8, 'mid': -6,
            'high_mid': -8, 'high': -12, 'air': -16,
        },
        'description': 'Heavy EDM festival mix with sub-bass emphasis',
    },
}


class ReferenceProfileManager:
    """Manages reference profiles for style matching."""

    def __init__(self, sample_rate: int = 48000, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.profiles: Dict[str, ReferenceProfile] = {}
        self._load_builtins()

    def _load_builtins(self):
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        for name, data in BUILTIN_PROFILES.items():
            band_defs = {
                'sub': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
                'mid': (500, 2000), 'high_mid': (2000, 4000),
                'high': (4000, 8000), 'air': (8000, 20000),
            }
            envelope = np.full_like(freqs, -40.0)
            for band_name, (lo, hi) in band_defs.items():
                mask = (freqs >= lo) & (freqs < hi)
                if np.any(mask):
                    envelope[mask] = data.get('band_energies', {}).get(
                        band_name, -20
                    )
            self.profiles[name] = ReferenceProfile(
                name=name, genre=data['genre'],
                spectral_envelope=envelope, frequencies=freqs,
                lufs_integrated=data['lufs_integrated'],
                lufs_range=data['lufs_range'],
                true_peak_db=data['true_peak_db'],
                dynamic_range_db=data['dynamic_range_db'],
                crest_factor_db=data['crest_factor_db'],
                band_energies=data.get('band_energies', {}),
                spectral_centroid=data.get('spectral_centroid', 0),
                spectral_rolloff=data.get('spectral_rolloff', 0),
                stereo_width=data.get('stereo_width', 0.5),
                description=data.get('description', ''),
            )

    def create_from_audio(self, name: str, audio: np.ndarray,
                          genre: str = 'unknown') -> ReferenceProfile:
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        n_frames = max(1, len(audio) // self.fft_size)
        avg_spec = np.zeros(self.fft_size // 2 + 1)
        window = np.hanning(self.fft_size)
        for i in range(n_frames):
            frame = audio[i * self.fft_size:(i + 1) * self.fft_size]
            if len(frame) < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - len(frame)))
            avg_spec += np.abs(np.fft.rfft(frame * window)) ** 2
        avg_spec = np.sqrt(avg_spec / n_frames)
        envelope_db = 20 * np.log10(avg_spec + 1e-10)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        lufs_est = 20 * np.log10(rms)
        peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-12)
        total_energy = np.sum(avg_spec)
        centroid = float(np.sum(freqs * avg_spec) / (total_energy + 1e-10))
        cumsum = np.cumsum(avg_spec)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        band_defs = {
            'sub': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
            'mid': (500, 2000), 'high_mid': (2000, 4000),
            'high': (4000, 8000), 'air': (8000, 20000),
        }
        band_energies = {}
        for band_name, (lo, hi) in band_defs.items():
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                band_energies[band_name] = float(
                    20 * np.log10(
                        np.sqrt(np.mean(avg_spec[mask] ** 2)) + 1e-10
                    )
                )
        profile = ReferenceProfile(
            name=name, genre=genre, spectral_envelope=envelope_db,
            frequencies=freqs, lufs_integrated=lufs_est, lufs_range=8.0,
            true_peak_db=peak_db,
            dynamic_range_db=abs(peak_db - lufs_est),
            crest_factor_db=abs(peak_db - lufs_est),
            band_energies=band_energies, spectral_centroid=centroid,
            spectral_rolloff=rolloff,
        )
        self.profiles[name] = profile
        return profile

    def get_profile(self, name: str) -> Optional[ReferenceProfile]:
        return self.profiles.get(name)

    def list_profiles(self) -> List[str]:
        return list(self.profiles.keys())

    def compute_distance(self, profile_a: str, profile_b: str) -> float:
        a = self.profiles.get(profile_a)
        b = self.profiles.get(profile_b)
        if a is None or b is None:
            return float('inf')
        spec_dist = np.sqrt(
            np.mean((a.spectral_envelope - b.spectral_envelope) ** 2)
        )
        lufs_dist = abs(a.lufs_integrated - b.lufs_integrated)
        dyn_dist = abs(a.dynamic_range_db - b.dynamic_range_db)
        return float(spec_dist * 0.5 + lufs_dist * 0.3 + dyn_dist * 0.2)
