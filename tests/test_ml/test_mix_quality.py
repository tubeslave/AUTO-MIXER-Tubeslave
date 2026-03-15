"""Tests for ml.mix_quality -- mix quality assessment metrics."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))


class TestMixQualityScore:
    """Tests for the MixQualityScore dataclass."""

    def test_score_creation(self):
        from ml.mix_quality import MixQualityScore
        score = MixQualityScore(
            overall=80.0, loudness=85.0, dynamics=75.0,
            spectral_balance=80.0, stereo_width=70.0,
            clarity=90.0, headroom=95.0, details={},
        )
        assert score.overall == 80.0
        assert score.loudness == 85.0
        assert score.headroom == 95.0

    def test_score_details_dict(self):
        from ml.mix_quality import MixQualityScore
        details = {'test_metric': 42.0}
        score = MixQualityScore(50, 50, 50, 50, 50, 50, 50, details)
        assert score.details['test_metric'] == 42.0

    def test_score_all_fields_present(self):
        from ml.mix_quality import MixQualityScore
        score = MixQualityScore(1, 2, 3, 4, 5, 6, 7, {})
        assert hasattr(score, 'overall')
        assert hasattr(score, 'loudness')
        assert hasattr(score, 'dynamics')
        assert hasattr(score, 'spectral_balance')
        assert hasattr(score, 'stereo_width')
        assert hasattr(score, 'clarity')
        assert hasattr(score, 'headroom')
        assert hasattr(score, 'details')


class TestMixQualityAnalyzer:
    """Tests for the MixQualityAnalyzer."""

    def test_instantiation(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        assert analyzer.sample_rate == 48000

    def test_analyze_empty_audio(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer()
        score = analyzer.analyze(np.array([]))
        assert score.overall == 0
        assert score.loudness == 0

    def test_analyze_returns_mix_quality_score(self):
        from ml.mix_quality import MixQualityAnalyzer, MixQualityScore
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        # Generate a simple sine wave mix
        t = np.linspace(0, 1.0, 48000)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        score = analyzer.analyze(audio, target_lufs=-18.0)
        assert isinstance(score, MixQualityScore)
        assert 0 <= score.overall <= 100

    def test_analyze_details_keys(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        t = np.linspace(0, 1.0, 48000)
        audio = 0.2 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        score = analyzer.analyze(audio)
        expected_keys = {
            'loudness_score', 'dynamics_score', 'spectral_balance_score',
            'stereo_width_score', 'clarity_score', 'headroom_score',
        }
        assert set(score.details.keys()) == expected_keys

    def test_analyze_headroom_high_for_quiet_signal(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        # Quiet signal has lots of headroom
        audio = np.random.randn(48000).astype(np.float32) * 0.01
        score = analyzer.analyze(audio)
        assert score.headroom >= 80.0

    def test_stereo_scoring_mono(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer()
        mono = np.random.randn(48000).astype(np.float32) * 0.1
        score = analyzer.analyze(mono)
        # Mono signals should get a mid-range stereo score (50)
        assert score.stereo_width == 50.0

    def test_stereo_scoring_stereo_input(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        # Create a stereo signal with some width
        t = np.linspace(0, 1.0, 48000)
        left = 0.3 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 440 * t + np.pi / 4)
        stereo = np.stack([left, right], axis=0)
        score = analyzer.analyze(stereo)
        assert score.stereo_width > 0

    def test_overall_is_weighted_average(self):
        from ml.mix_quality import MixQualityAnalyzer
        analyzer = MixQualityAnalyzer(sample_rate=48000)
        t = np.linspace(0, 1.0, 48000)
        audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        score = analyzer.analyze(audio)
        expected_overall = (
            score.loudness * 0.2 + score.dynamics * 0.2 +
            score.spectral_balance * 0.2 + score.stereo_width * 0.1 +
            score.clarity * 0.2 + score.headroom * 0.1
        )
        assert abs(score.overall - expected_overall) < 1e-3
