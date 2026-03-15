"""
Mix style transfer — applies the style of a reference mix to the current mix.
Generates EQ, dynamics, and gain adjustments to match a target style.
"""
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class StyleTransferResult:
    """Result of style transfer computation."""
    gain_adjustment_db: float = 0.0
    eq_corrections: List[Dict] = field(default_factory=list)
    comp_threshold_db: float = -20.0
    comp_ratio: float = 2.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0
    stereo_width_target: float = 0.5
    confidence: float = 0.5
    notes: List[str] = field(default_factory=list)

class MixStyleTransfer:
    """Computes parameter adjustments to match a target mix style."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.band_defs = {
            'sub': (20, 60), 'bass': (60, 250), 'low_mid': (250, 500),
            'mid': (500, 2000), 'high_mid': (2000, 4000),
            'high': (4000, 8000), 'air': (8000, 20000)
        }

    def compute_transfer(self, current_style, target_style,
                        max_eq_db: float = 8.0, max_gain_db: float = 6.0) -> StyleTransferResult:
        """Compute parameter adjustments to transfer from current to target style."""
        result = StyleTransferResult()

        # Gain adjustment
        gain_diff = target_style.loudness_lufs - current_style.loudness_lufs
        result.gain_adjustment_db = float(np.clip(gain_diff, -max_gain_db, max_gain_db))

        # EQ corrections per band
        eq_corrections = []
        for band_name, (lo, hi) in self.band_defs.items():
            current_level = current_style.band_levels.get(band_name, -30)
            target_level = target_style.band_levels.get(band_name, -30)
            diff = target_level - current_level
            diff = float(np.clip(diff, -max_eq_db, max_eq_db))

            if abs(diff) > 0.5:
                center_freq = np.sqrt(lo * hi)
                q = center_freq / (hi - lo) * 1.5
                eq_type = 'peaking'
                if band_name == 'sub':
                    eq_type = 'low_shelf'
                elif band_name == 'air':
                    eq_type = 'high_shelf'

                eq_corrections.append({
                    'band': band_name,
                    'frequency': float(center_freq),
                    'gain_db': round(diff, 1),
                    'q': round(float(q), 2),
                    'type': eq_type,
                })
        result.eq_corrections = eq_corrections

        # Dynamics matching
        target_dr = target_style.dynamic_range_db
        current_dr = current_style.dynamic_range_db

        if current_dr > target_dr + 3:
            # Need more compression
            result.comp_ratio = min(8.0, 2.0 + (current_dr - target_dr) / 6)
            result.comp_threshold_db = target_style.loudness_lufs + 6
            result.notes.append(f"Increase compression: current DR={current_dr:.1f}dB, target={target_dr:.1f}dB")
        elif current_dr < target_dr - 3:
            # Need less compression
            result.comp_ratio = max(1.0, 2.0 - (target_dr - current_dr) / 10)
            result.comp_threshold_db = target_style.loudness_lufs + 12
            result.notes.append(f"Reduce compression: current DR={current_dr:.1f}dB, target={target_dr:.1f}dB")
        else:
            result.comp_ratio = 2.0
            result.comp_threshold_db = target_style.loudness_lufs + 8

        # Attack/release from target
        result.comp_attack_ms = target_style.avg_attack_ms
        result.comp_release_ms = target_style.avg_release_ms

        # Stereo width
        result.stereo_width_target = target_style.stereo_width

        # Confidence based on how different the styles are
        try:
            from neural_mix_extractor import NeuralMixExtractor
            extractor = NeuralMixExtractor(self.sample_rate)
            distance = extractor.compute_distance(current_style, target_style)
            result.confidence = max(0.2, min(0.95, 1.0 - distance / 20.0))
        except Exception:
            result.confidence = 0.6

        if result.gain_adjustment_db != 0:
            result.notes.append(f"Gain: {result.gain_adjustment_db:+.1f}dB")
        if eq_corrections:
            result.notes.append(f"EQ: {len(eq_corrections)} band corrections")

        return result

    def apply_to_channels(self, transfer_result: StyleTransferResult,
                          channel_instruments: Dict[int, str]) -> Dict[int, Dict]:
        """Generate per-channel adjustments based on style transfer."""
        channel_adjustments = {}

        # Instrument-specific scaling
        scaling = {
            'lead_vocal': {'gain': 1.0, 'eq': 0.7, 'comp': 0.8},
            'kick': {'gain': 1.0, 'eq': 1.0, 'comp': 1.0},
            'snare': {'gain': 1.0, 'eq': 0.9, 'comp': 1.0},
            'bass_guitar': {'gain': 1.0, 'eq': 0.8, 'comp': 0.9},
            'electric_guitar': {'gain': 0.8, 'eq': 1.0, 'comp': 0.7},
            'acoustic_guitar': {'gain': 0.8, 'eq': 0.9, 'comp': 0.6},
            'keys_piano': {'gain': 0.7, 'eq': 0.8, 'comp': 0.5},
            'overheads': {'gain': 0.6, 'eq': 0.7, 'comp': 0.4},
        }

        default_scale = {'gain': 0.7, 'eq': 0.6, 'comp': 0.5}

        for ch_id, instrument in channel_instruments.items():
            scale = scaling.get(instrument, default_scale)

            ch_adj = {
                'gain_db': transfer_result.gain_adjustment_db * scale['gain'],
                'eq_bands': [],
                'comp_threshold': transfer_result.comp_threshold_db,
                'comp_ratio': 1.0 + (transfer_result.comp_ratio - 1.0) * scale['comp'],
                'comp_attack_ms': transfer_result.comp_attack_ms,
                'comp_release_ms': transfer_result.comp_release_ms,
            }

            for eq in transfer_result.eq_corrections:
                ch_adj['eq_bands'].append({
                    'frequency': eq['frequency'],
                    'gain_db': eq['gain_db'] * scale['eq'],
                    'q': eq['q'],
                    'type': eq['type'],
                })

            channel_adjustments[ch_id] = ch_adj

        return channel_adjustments
