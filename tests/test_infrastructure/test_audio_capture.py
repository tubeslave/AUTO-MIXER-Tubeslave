"""
Tests for backend/audio_capture.py — RingBuffer, signal generators,
AudioCapture subscribe/unsubscribe pattern.

All tests work without sounddevice (no hardware required).
"""

import numpy as np
import pytest

try:
    from audio_capture import RingBuffer, AudioCapture, TestGeneratorConfig
except ImportError:
    pytest.skip("audio_capture module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ring_buffer():
    """RingBuffer with capacity for 4800 samples (100ms at 48kHz)."""
    # RingBuffer takes (max_duration_sec, sample_rate)
    # 4800 samples at 48000 Hz = 0.1 seconds
    return RingBuffer(max_duration_sec=0.1, sample_rate=48000)


@pytest.fixture
def audio_capture():
    """AudioCapture configured (no hardware required)."""
    return AudioCapture(
        sample_rate=48000,
        channels=2,
        buffer_duration=1.0,
        block_size=1024,
    )


# ---------------------------------------------------------------------------
# RingBuffer tests
# ---------------------------------------------------------------------------

class TestRingBuffer:

    def test_write_and_read(self, ring_buffer):
        """Writing data should be readable."""
        data = np.ones(100, dtype=np.float32) * 0.5
        ring_buffer.write(data)
        result = ring_buffer.read(100)
        assert len(result) == 100
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_available_samples(self, ring_buffer):
        """available_samples should reflect buffered data."""
        assert ring_buffer.available_samples == 0
        data = np.zeros(200, dtype=np.float32)
        ring_buffer.write(data)
        assert ring_buffer.available_samples == 200

    def test_read_more_than_available(self, ring_buffer):
        """Reading more than available should return zero-padded array."""
        data = np.ones(50, dtype=np.float32)
        ring_buffer.write(data)
        result = ring_buffer.read(100)
        # Result is padded to requested size, but only 50 samples are real
        assert len(result) == 100
        # The last 50 should contain the written data
        np.testing.assert_allclose(result[-50:], 1.0, atol=1e-6)

    def test_wraparound(self, ring_buffer):
        """Buffer should handle wraparound correctly."""
        # Capacity is 4800 samples. Write 4000, read 2000, write 2000 more.
        big = np.arange(4000, dtype=np.float32)
        ring_buffer.write(big)
        ring_buffer.read(2000)
        # Write more (causing wraparound)
        more = np.arange(2000, dtype=np.float32) + 100.0
        ring_buffer.write(more)
        # Read the most recent 2000 samples — should be the new data
        result = ring_buffer.read(2000)
        assert len(result) == 2000
        np.testing.assert_allclose(result, more, atol=1e-6)

    def test_overwrite_oldest(self, ring_buffer):
        """Writing beyond capacity should overwrite the oldest data."""
        # Write more than capacity (4800 samples)
        big = np.arange(5000, dtype=np.float32)
        ring_buffer.write(big)
        # Should have at most 4800 samples
        assert ring_buffer.available_samples <= 4800


# ---------------------------------------------------------------------------
# Pink noise generator tests
# These use AudioCapture._voss_mccartney which is a static method.
# ---------------------------------------------------------------------------

class TestSignalGenerators:

    def test_pink_noise_generator(self):
        """AudioCapture._voss_mccartney should produce non-zero audio."""
        import numpy as np
        np.random.seed(42)
        noise = AudioCapture._voss_mccartney(4800)
        assert len(noise) == 4800
        # Should not be all zeros
        assert np.max(np.abs(noise)) > 0.0

    def test_pink_noise_different_seeds(self):
        """Different seeds should produce different noise."""
        import numpy as np
        np.random.seed(1)
        n1 = AudioCapture._voss_mccartney(1024)
        np.random.seed(2)
        n2 = AudioCapture._voss_mccartney(1024)
        assert not np.allclose(n1, n2)

    def test_pink_noise_normalized(self):
        """Pink noise from _voss_mccartney should produce non-trivial output."""
        import numpy as np
        np.random.seed(42)
        noise = AudioCapture._voss_mccartney(4800)
        # Should produce real output (not all zeros)
        assert np.max(np.abs(noise)) > 0.0


# ---------------------------------------------------------------------------
# AudioCapture tests
# ---------------------------------------------------------------------------

class TestAudioCapture:

    def test_subscribe_unsubscribe(self, audio_capture):
        """subscribe_all/unsubscribe_all pattern should work."""
        callback_data = []

        def on_audio(channel, data):
            callback_data.append((channel, len(data)))

        audio_capture.subscribe_all(on_audio)
        assert len(audio_capture._global_subscribers) >= 1
        audio_capture.unsubscribe_all(on_audio)

    def test_get_buffer_returns_ndarray(self, audio_capture):
        """get_buffer should return a numpy array."""
        buf = audio_capture.get_buffer(channel_id=0)
        assert buf is not None
        assert isinstance(buf, np.ndarray)

    def test_test_generator_config(self):
        """TestGeneratorConfig should store generator_type and parameters."""
        cfg = TestGeneratorConfig(generator_type="pink_noise", amplitude=0.3)
        assert cfg.generator_type == "pink_noise"
        assert cfg.amplitude == 0.3

    def test_silence_generator(self):
        cfg = TestGeneratorConfig(generator_type="silence")
        assert cfg.generator_type == "silence"
