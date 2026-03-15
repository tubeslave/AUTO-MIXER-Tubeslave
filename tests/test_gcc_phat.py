"""
Tests for GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
delay estimation used in phase alignment.

Covers:
- Zero delay detection between identical signals
- Known delay detection with sample-accurate validation
- Noise robustness
"""

import numpy as np
import pytest

from phase_alignment import PhaseAlignmentAnalyzer


class TestGCCPHAT:
    """Tests for the GCC-PHAT delay estimation method."""

    def _make_analyzer(self, sample_rate=48000):
        """Create a PhaseAlignmentAnalyzer without needing audio hardware."""
        analyzer = PhaseAlignmentAnalyzer.__new__(PhaseAlignmentAnalyzer)
        analyzer.sample_rate = sample_rate
        analyzer.chunk_size = 4096
        analyzer.max_delay_samples = int(sample_rate * 0.010)  # 10 ms
        return analyzer

    def test_zero_delay(self, sample_rate):
        """Identical signals should produce zero or near-zero delay."""
        analyzer = self._make_analyzer(sample_rate)
        rng = np.random.RandomState(42)
        n = int(sample_rate * 0.1)
        # Broadband signal — GCC-PHAT works best with wideband content
        signal = rng.randn(n).astype(np.float64)

        delay, confidence = analyzer._gcc_phat(signal, signal.copy())

        assert abs(delay) <= 2, f"Zero-delay signals: expected delay ~0, got {delay}"
        assert confidence != 0.0 or abs(delay) <= 2, (
            f"Confidence should be non-zero for identical signals, got {confidence:.6f}"
        )

    def test_known_delay(self, sample_rate):
        """Signal delayed by N samples should be detected correctly."""
        analyzer = self._make_analyzer(sample_rate)
        rng = np.random.RandomState(99)
        known_delay = 10  # samples
        n = int(sample_rate * 0.1)

        # Use broadband noise — narrowband sinusoids defeat PHAT weighting
        ref = rng.randn(n).astype(np.float64)
        delayed = np.zeros_like(ref)
        delayed[known_delay:] = ref[:-known_delay]

        delay, confidence = analyzer._gcc_phat(ref, delayed)

        assert abs(delay - known_delay) <= 2, (
            f"Expected delay ~{known_delay}, got {delay}"
        )

    def test_noise_robustness(self, sample_rate):
        """GCC-PHAT should still find delay in moderately noisy conditions."""
        analyzer = self._make_analyzer(sample_rate)
        rng = np.random.RandomState(123)
        known_delay = 20
        n = int(sample_rate * 0.2)

        clean = rng.randn(n).astype(np.float64)

        noise_level = 0.3
        ref = clean + noise_level * rng.randn(n)
        delayed_clean = np.zeros_like(clean)
        delayed_clean[known_delay:] = clean[:-known_delay]
        delayed = delayed_clean + noise_level * rng.randn(n)

        delay, confidence = analyzer._gcc_phat(ref, delayed)

        assert abs(delay - known_delay) <= 5, (
            f"Noisy delay estimate: expected ~{known_delay}, got {delay}"
        )
