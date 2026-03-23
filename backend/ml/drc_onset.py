"""
DRC onset detection -- detects compression onset for adaptive attack/release.
Based on Giannoulis et al. "Digital Dynamic Range Compressor Design" (2012).
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OnsetEvent:
    """Detected onset event."""
    time_sec: float
    strength: float  # 0-1 onset strength
    type: str  # 'transient', 'gradual', 'sustained'
    recommended_attack_ms: float
    recommended_release_ms: float


class DRCOnsetDetector:
    """Detects signal onsets for adaptive compressor timing."""

    def __init__(self, sample_rate: int = 48000, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.hop_time = block_size / sample_rate
        self._prev_energy = 0.0
        self._energy_history: List[float] = []
        self._onset_history: List[OnsetEvent] = []
        self._time = 0.0
        self._smooth_energy = 0.0
        self._alpha = 0.3

    def process(self, samples: np.ndarray) -> Optional[OnsetEvent]:
        energy = float(np.mean(samples ** 2))
        energy_db = 10 * np.log10(energy + 1e-12)
        self._smooth_energy = (self._alpha * energy +
                               (1 - self._alpha) * self._smooth_energy)
        delta = energy_db - (10 * np.log10(self._prev_energy + 1e-12))
        self._energy_history.append(energy_db)
        if len(self._energy_history) > 100:
            self._energy_history.pop(0)
        self._prev_energy = energy
        self._time += self.hop_time
        onset = None
        if delta > 6.0:
            strength = min(1.0, delta / 20.0)
            if delta > 12.0:
                onset_type = 'transient'
            elif delta > 8.0:
                onset_type = 'gradual'
            else:
                onset_type = 'sustained'
            attack_ms, release_ms = self._recommend_timing(onset_type, strength)
            onset = OnsetEvent(
                time_sec=self._time,
                strength=strength,
                type=onset_type,
                recommended_attack_ms=attack_ms,
                recommended_release_ms=release_ms,
            )
            self._onset_history.append(onset)
            if len(self._onset_history) > 50:
                self._onset_history.pop(0)
        return onset

    def _recommend_timing(self, onset_type: str,
                          strength: float) -> Tuple[float, float]:
        if onset_type == 'transient':
            attack = max(0.1, 5.0 * (1 - strength))
            release = max(50.0, 200.0 * (1 - strength))
        elif onset_type == 'gradual':
            attack = max(5.0, 20.0 * (1 - strength))
            release = max(100.0, 500.0 * (1 - strength))
        else:
            attack = max(10.0, 50.0 * (1 - strength))
            release = max(200.0, 1000.0 * (1 - strength))
        return attack, release

    def get_adaptive_params(self) -> dict:
        if not self._onset_history:
            return {'attack_ms': 10.0, 'release_ms': 100.0, 'type': 'default'}
        recent = self._onset_history[-5:]
        avg_attack = np.mean([o.recommended_attack_ms for o in recent])
        avg_release = np.mean([o.recommended_release_ms for o in recent])
        dominant_type = max(
            set(o.type for o in recent),
            key=lambda t: sum(1 for o in recent if o.type == t)
        )
        return {
            'attack_ms': float(avg_attack),
            'release_ms': float(avg_release),
            'type': dominant_type,
        }

    def get_density(self) -> float:
        if self._time < 1.0:
            return 0.0
        window = 2.0
        recent = [o for o in self._onset_history
                  if o.time_sec > self._time - window]
        return len(recent) / window

    def reset(self):
        self._prev_energy = 0.0
        self._energy_history.clear()
        self._onset_history.clear()
        self._time = 0.0
        self._smooth_energy = 0.0
