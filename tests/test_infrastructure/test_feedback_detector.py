"""
Tests for backend/feedback_detector.py — FeedbackDetector, FeedbackDetectorConfig,
_freq_close, _q_for_feedback, NotchFilter, _ChannelFeedbackState, and
processing synthetic signals.

All tests work without hardware or network.
"""

import numpy as np
import pytest

try:
    from feedback_detector import (
        FeedbackDetector,
        FeedbackDetectorConfig,
        FeedbackEvent,
        NotchFilter,
        _ChannelFeedbackState,
        _PeakState,
        _freq_close,
        _q_for_feedback,
        _find_spectral_peaks,
        MAX_NOTCH_FILTERS,
        MAX_NOTCH_DEPTH_DB,
    )
except ImportError:
    pytest.skip("feedback_detector module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Default FeedbackDetectorConfig."""
    return FeedbackDetectorConfig()


@pytest.fixture
def detector():
    """FeedbackDetector with default config."""
    return FeedbackDetector()


@pytest.fixture
def custom_detector():
    """FeedbackDetector with relaxed thresholds for easier triggering."""
    cfg = FeedbackDetectorConfig(
        persistence_frames=2,
        min_confidence=0.1,
        peak_height_db=-40.0,
        peak_prominence_db=3.0,
    )
    return FeedbackDetector(config=cfg)


def _make_sine(freq_hz, duration_sec=0.1, sample_rate=48000, amplitude=0.8):
    """Generate a sine wave."""
    t = np.arange(int(sample_rate * duration_sec)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _make_feedback_signal(freq_hz, num_blocks=20, block_size=2048,
                          sample_rate=48000, amplitude=0.9):
    """Generate a sustained, growing sine (simulated feedback)."""
    blocks = []
    for i in range(num_blocks):
        t = np.arange(block_size) / sample_rate
        # Growing amplitude to simulate feedback
        amp = min(amplitude * (1.0 + i * 0.1), 1.0)
        block = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# _freq_close tests
# ---------------------------------------------------------------------------

class TestFreqClose:

    def test_identical_frequencies(self):
        assert _freq_close(1000.0, 1000.0) == True

    def test_close_frequencies(self):
        # 1000 Hz and 1010 Hz are about 17 cents apart
        assert _freq_close(1000.0, 1010.0, tolerance_cents=100.0) == True

    def test_far_frequencies(self):
        # 1000 Hz and 2000 Hz are 1200 cents apart
        assert _freq_close(1000.0, 2000.0, tolerance_cents=100.0) == False

    def test_zero_frequency(self):
        assert _freq_close(0.0, 1000.0) == False
        assert _freq_close(1000.0, 0.0) == False

    def test_negative_frequency(self):
        assert _freq_close(-100.0, 100.0) == False


# ---------------------------------------------------------------------------
# _q_for_feedback tests
# ---------------------------------------------------------------------------

class TestQForFeedback:

    def test_high_confidence(self):
        q = _q_for_feedback(1.0)
        assert q >= 8.0

    def test_low_confidence(self):
        q = _q_for_feedback(0.0)
        assert q >= 4.0

    def test_mid_confidence(self):
        q = _q_for_feedback(0.5)
        assert 4.0 <= q <= 10.0

    def test_clamped(self):
        q = _q_for_feedback(2.0)
        assert q <= 10.0


# ---------------------------------------------------------------------------
# FeedbackDetectorConfig tests
# ---------------------------------------------------------------------------

class TestFeedbackDetectorConfig:

    def test_default_values(self, config):
        assert config.sample_rate == 48000
        assert config.fft_size == 2048
        assert config.persistence_frames == 6
        assert config.min_confidence == 0.5

    def test_custom_values(self):
        cfg = FeedbackDetectorConfig(
            sample_rate=44100,
            fft_size=4096,
            persistence_frames=10,
        )
        assert cfg.sample_rate == 44100
        assert cfg.fft_size == 4096
        assert cfg.persistence_frames == 10


# ---------------------------------------------------------------------------
# NotchFilter tests
# ---------------------------------------------------------------------------

class TestNotchFilter:

    def test_creation(self):
        nf = NotchFilter(frequency=1000.0, gain_db=-6.0, q_factor=8.0)
        assert nf.frequency == 1000.0
        assert nf.gain_db == -6.0
        assert nf.q_factor == 8.0
        assert nf.slot_index == 0


# ---------------------------------------------------------------------------
# _PeakState tests
# ---------------------------------------------------------------------------

class TestPeakState:

    def test_is_growing_empty_history(self):
        peak = _PeakState(frequency=1000.0, magnitude_db=-10.0)
        assert peak.is_growing() is False

    def test_is_growing_with_rising_history(self):
        peak = _PeakState(
            frequency=1000.0,
            magnitude_db=-5.0,
            magnitude_history=[-15.0, -12.0, -10.0, -8.0, -5.0],
        )
        assert peak.is_growing() is True

    def test_is_growing_with_falling_history(self):
        peak = _PeakState(
            frequency=1000.0,
            magnitude_db=-15.0,
            magnitude_history=[-5.0, -8.0, -10.0, -12.0, -15.0],
        )
        assert peak.is_growing() is False

    def test_age_sec(self):
        peak = _PeakState(
            frequency=1000.0,
            magnitude_db=-10.0,
            first_seen=100.0,
            last_seen=102.5,
        )
        assert abs(peak.age_sec - 2.5) < 1e-6


# ---------------------------------------------------------------------------
# _ChannelFeedbackState tests
# ---------------------------------------------------------------------------

class TestChannelFeedbackState:

    def test_add_notch(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        nf = state.add_notch(1000.0, 0.8, now=0.0)
        assert nf is not None
        assert nf.frequency == 1000.0
        assert len(state.notch_filters) == 1

    def test_add_notch_deepens_existing(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        nf1 = state.add_notch(1000.0, 0.8, now=0.0)
        initial_gain = nf1.gain_db
        nf2 = state.add_notch(1000.0, 0.9, now=1.0)
        assert nf2.gain_db <= initial_gain
        assert len(state.notch_filters) == 1  # Same filter deepened

    def test_max_notch_limit(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        for i in range(MAX_NOTCH_FILTERS):
            freq = 500.0 + i * 500.0  # Different enough frequencies
            nf = state.add_notch(freq, 0.8, now=0.0)
            assert nf is not None
        # One more should return None
        nf = state.add_notch(10000.0, 0.8, now=0.0)
        assert nf is None

    def test_release_notch(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        state.add_notch(1000.0, 0.8, now=0.0)
        assert state.release_notch(0) is True
        assert len(state.notch_filters) == 0

    def test_release_nonexistent_notch(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        assert state.release_notch(5) is False

    def test_fader_reduction(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        reduction = state.apply_fader_reduction(step_db=-3.0, floor_db=-18.0)
        assert reduction == -3.0
        reduction = state.apply_fader_reduction(step_db=-3.0, floor_db=-18.0)
        assert reduction == -6.0

    def test_fader_reduction_floor(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        for _ in range(20):
            state.apply_fader_reduction(step_db=-3.0, floor_db=-18.0)
        assert state.fader_reduction_db >= -18.0

    def test_get_stale_notches(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        state.add_notch(1000.0, 0.8, now=0.0)
        # With now far in the future and no tracked peaks
        stale = state.get_stale_notches(now=100.0, max_age_sec=30.0)
        assert len(stale) == 1

    def test_get_stale_notches_none_stale(self, config):
        state = _ChannelFeedbackState(channel_id=1, config=config)
        state.add_notch(1000.0, 0.8, now=0.0)
        # Recent notch should not be stale
        stale = state.get_stale_notches(now=1.0, max_age_sec=30.0)
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# FeedbackDetector API tests
# ---------------------------------------------------------------------------

class TestFeedbackDetectorAPI:

    def test_process_audio_returns_none_for_silence(self, detector):
        silence = np.zeros(2048, dtype=np.float32)
        event = detector.process_audio(1, silence)
        assert event is None

    def test_process_audio_returns_none_for_noise(self, detector):
        np.random.seed(42)
        noise = np.random.randn(2048).astype(np.float32) * 0.01
        event = detector.process_audio(1, noise)
        assert event is None

    def test_get_active_notches_empty(self, detector):
        notches = detector.get_active_notches(1)
        assert notches == []

    def test_get_fader_reduction_default(self, detector):
        assert detector.get_fader_reduction(1) == 0.0

    def test_reset_channel(self, detector):
        # Process some audio to create state
        silence = np.zeros(2048, dtype=np.float32)
        detector.process_audio(1, silence)
        detector.reset_channel(1)
        assert detector.get_channel_state(1) is None

    def test_reset_all(self, detector):
        silence = np.zeros(2048, dtype=np.float32)
        detector.process_audio(1, silence)
        detector.process_audio(2, silence)
        detector.reset_all()
        assert detector.get_channel_state(1) is None
        assert detector.get_channel_state(2) is None

    def test_get_diagnostics_empty(self, detector):
        diag = detector.get_diagnostics(1)
        assert diag["tracked_peaks"] == 0
        assert diag["fader_reduction_db"] == 0.0

    def test_get_diagnostics_after_processing(self, detector):
        silence = np.zeros(2048, dtype=np.float32)
        detector.process_audio(1, silence)
        diag = detector.get_diagnostics(1)
        assert "channel_id" in diag
        assert diag["channel_id"] == 1


# ---------------------------------------------------------------------------
# FeedbackDetector OSC command generation tests
# ---------------------------------------------------------------------------

class TestFeedbackDetectorOSCCommands:

    def test_generate_osc_commands_empty(self, detector):
        commands = detector.generate_osc_commands(1)
        assert commands == []

    def test_generate_osc_commands_with_notch(self, config):
        detector = FeedbackDetector(config=config)
        state = detector._get_or_create_state(1)
        state.add_notch(1000.0, 0.8, now=0.0)
        commands = detector.generate_osc_commands(1)
        assert len(commands) > 0
        # Should include eq/on and band frequency/gain/q
        addresses = [c[0] for c in commands]
        assert any("eq/on" in a for a in addresses)
        assert any("1f" in a for a in addresses)
        assert any("1g" in a for a in addresses)

    def test_generate_reset_osc_commands(self, detector):
        commands = detector.generate_reset_osc_commands(1)
        assert len(commands) > 0
        # Should reset main EQ bands 1-4 and pre-EQ bands 1-3
        addresses = [c[0] for c in commands]
        assert any("eq/1g" in a for a in addresses)
        assert any("peq/1g" in a for a in addresses)


# ---------------------------------------------------------------------------
# _find_spectral_peaks tests
# ---------------------------------------------------------------------------

class TestFindSpectralPeaks:

    def test_finds_peak_in_synthetic_signal(self):
        """A strong tone should produce a detectable spectral peak."""
        sr = 48000
        fft_size = 2048
        freq = 1000.0
        t = np.arange(fft_size) / sr
        signal = 0.9 * np.sin(2 * np.pi * freq * t)
        window = np.hanning(fft_size)
        spectrum = np.abs(np.fft.rfft(signal * window))
        mag_db = 20.0 * np.log10(spectrum + 1e-10)
        peaks, props = _find_spectral_peaks(
            mag_db, height=-30.0, distance=5, prominence=6.0
        )
        assert len(peaks) > 0

    def test_no_peaks_in_silence(self):
        silence_db = np.full(1025, -80.0)
        peaks, props = _find_spectral_peaks(
            silence_db, height=-30.0, distance=5, prominence=6.0
        )
        assert len(peaks) == 0
