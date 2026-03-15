"""
Tests for the FeedbackDetector (real-time acoustic feedback detection).

Covers:
- No false positives on silence
- Detection of sustained tones (simulated feedback)
- Maximum notch filter limit enforcement
- OSC command generation for Wing mixer
"""

import numpy as np
import pytest

from feedback_detector import (
    FeedbackDetector,
    FeedbackDetectorConfig,
    FeedbackEvent,
    NotchFilter,
    MAX_NOTCH_FILTERS,
    MAX_NOTCH_DEPTH_DB,
    _ChannelFeedbackState,
)


class TestNoFeedbackOnSilence:
    """Silence should never trigger feedback detection."""

    def test_no_feedback_on_silence(self, sample_rate):
        """Processing silence should return None (no feedback event)."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        silence = np.zeros(config.fft_size * 4, dtype=np.float32)
        event = detector.process_audio(channel_id=1, samples=silence)

        assert event is None, "Silence should not trigger feedback detection"

    def test_no_notches_after_silence(self, sample_rate):
        """After processing silence, no notch filters should be active."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        silence = np.zeros(config.fft_size * 4, dtype=np.float32)
        detector.process_audio(channel_id=1, samples=silence)

        notches = detector.get_active_notches(channel_id=1)
        assert len(notches) == 0, "No notches should be active after silence"

    def test_no_fader_reduction_on_silence(self, sample_rate):
        """Fader reduction should be 0.0 after processing silence."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        silence = np.zeros(config.fft_size * 4, dtype=np.float32)
        detector.process_audio(channel_id=1, samples=silence)

        reduction = detector.get_fader_reduction(channel_id=1)
        assert reduction == 0.0, f"Fader reduction should be 0, got {reduction}"


class TestDetectsSustainedTone:
    """A persistent loud tone should eventually be classified as feedback."""

    def _make_sustained_tone(self, freq_hz, sample_rate, duration_sec=0.5,
                             amplitude=0.9):
        """Generate a sustained pure tone that mimics feedback."""
        n = int(sample_rate * duration_sec)
        t = np.arange(n, dtype=np.float32) / sample_rate
        return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    def test_detects_sustained_tone(self, sample_rate):
        """A persistent loud 3 kHz tone should eventually trigger detection."""
        config = FeedbackDetectorConfig(
            sample_rate=sample_rate,
            persistence_frames=3,  # Lower threshold for faster test
            min_confidence=0.3,
            peak_height_db=-40.0,
            peak_prominence_db=4.0,
        )
        detector = FeedbackDetector(config)

        # Feed many blocks of a sustained tone
        tone = self._make_sustained_tone(3000.0, sample_rate, duration_sec=2.0)
        block_size = config.fft_size
        events = []

        for start in range(0, len(tone) - block_size, block_size):
            block = tone[start:start + block_size]
            event = detector.process_audio(channel_id=1, samples=block)
            if event is not None:
                events.append(event)

        # We expect at least one feedback event
        assert len(events) > 0, (
            "Sustained 3 kHz tone should trigger at least one feedback event"
        )

        # The detected frequency should be near 3 kHz
        first_event = events[0]
        assert isinstance(first_event, FeedbackEvent)
        assert abs(first_event.frequency_hz - 3000.0) < 200.0, (
            f"Detected frequency {first_event.frequency_hz} should be near 3000 Hz"
        )

    def test_feedback_event_fields(self, sample_rate):
        """FeedbackEvent should have all expected fields."""
        event = FeedbackEvent(
            channel_id=1,
            frequency_hz=2500.0,
            gain_db=-6.0,
            confidence=0.85,
            action="notch",
        )
        assert event.channel_id == 1
        assert event.frequency_hz == 2500.0
        assert event.gain_db == -6.0
        assert event.confidence == 0.85
        assert event.action == "notch"
        assert event.timestamp > 0


class TestMaxNotchLimit:
    """The detector should respect MAX_NOTCH_FILTERS per channel."""

    def test_max_notch_limit(self):
        """_ChannelFeedbackState should not add more than MAX_NOTCH_FILTERS notches."""
        config = FeedbackDetectorConfig()
        state = _ChannelFeedbackState(channel_id=1, config=config)

        import time
        now = time.monotonic()

        # Fill up all notch slots
        for i in range(MAX_NOTCH_FILTERS):
            freq = 1000.0 + i * 500.0
            nf = state.add_notch(freq, confidence=0.9, now=now)
            assert nf is not None, f"Notch {i} should have been created"

        assert len(state.notch_filters) == MAX_NOTCH_FILTERS

        # Trying to add one more should return None
        extra = state.add_notch(9000.0, confidence=0.9, now=now)
        assert extra is None, "Should not exceed MAX_NOTCH_FILTERS"
        assert len(state.notch_filters) == MAX_NOTCH_FILTERS

    def test_notch_filter_dataclass(self):
        """NotchFilter should store all required fields."""
        nf = NotchFilter(
            frequency=2000.0,
            gain_db=-6.0,
            q_factor=8.0,
            slot_index=0,
            applied_at=100.0,
        )
        assert nf.frequency == 2000.0
        assert nf.gain_db == -6.0
        assert nf.q_factor == 8.0
        assert nf.slot_index == 0

    def test_notch_release(self):
        """release_notch should remove a notch and re-index remaining slots."""
        config = FeedbackDetectorConfig()
        state = _ChannelFeedbackState(channel_id=1, config=config)

        import time
        now = time.monotonic()

        # Add 3 notches
        for i in range(3):
            state.add_notch(1000.0 + i * 1000.0, confidence=0.8, now=now)

        assert len(state.notch_filters) == 3

        # Release the middle one (slot_index=1)
        result = state.release_notch(1)
        assert result is True
        assert len(state.notch_filters) == 2

        # Slots should be re-indexed 0, 1
        for i, nf in enumerate(state.notch_filters):
            assert nf.slot_index == i

    def test_stale_notch_detection(self):
        """get_stale_notches should return old notches whose frequency is no longer tracked."""
        config = FeedbackDetectorConfig()
        state = _ChannelFeedbackState(channel_id=1, config=config)

        import time
        old_time = time.monotonic() - 60.0  # 60 seconds ago
        now = time.monotonic()

        # Add a notch that was applied a long time ago
        nf = NotchFilter(
            frequency=4000.0, gain_db=-6.0, q_factor=8.0,
            slot_index=0, applied_at=old_time,
        )
        state.notch_filters.append(nf)

        # No tracked peaks near 4000 Hz
        stale = state.get_stale_notches(now, max_age_sec=30.0)
        assert len(stale) == 1
        assert stale[0].frequency == 4000.0


class TestOSCCommandGeneration:
    """Tests for generating Wing OSC commands from feedback state."""

    def test_osc_command_generation(self, sample_rate):
        """generate_osc_commands should produce valid OSC address/value pairs."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        # Manually inject a notch into channel state
        state = detector._get_or_create_state(channel_id=1)
        import time
        now = time.monotonic()
        state.add_notch(2000.0, confidence=0.9, now=now)

        commands = detector.generate_osc_commands(channel_id=1)

        assert len(commands) > 0, "Should generate at least one OSC command"

        # Check command structure
        for addr, value in commands:
            assert isinstance(addr, str)
            assert addr.startswith("/ch/1/")

        # Should include an EQ on command
        addresses = [addr for addr, _ in commands]
        assert "/ch/1/eq/on" in addresses

    def test_osc_commands_empty_for_unknown_channel(self, sample_rate):
        """generate_osc_commands for an unknown channel should return empty list."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        commands = detector.generate_osc_commands(channel_id=99)
        assert commands == []

    def test_reset_osc_commands(self, sample_rate):
        """generate_reset_osc_commands should zero out all notch bands."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        commands = detector.generate_reset_osc_commands(channel_id=1)
        assert len(commands) > 0

        # All gain commands should set to 0.0
        for addr, value in commands:
            if addr.endswith("g"):
                assert value == 0.0, f"Reset gain for {addr} should be 0.0"

    def test_reset_channel(self, sample_rate):
        """reset_channel should clear all state for that channel."""
        config = FeedbackDetectorConfig(sample_rate=sample_rate)
        detector = FeedbackDetector(config)

        # Create some state
        state = detector._get_or_create_state(channel_id=5)
        import time
        state.add_notch(1000.0, confidence=0.8, now=time.monotonic())

        assert len(detector.get_active_notches(5)) == 1

        # Reset
        detector.reset_channel(5)
        assert detector.get_active_notches(5) == []
        assert detector.get_fader_reduction(5) == 0.0
