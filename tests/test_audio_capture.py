"""
Tests for audio capture utilities (ring buffer, subscriptions, test generators).

Since there is no standalone audio_capture.py module, these tests validate
the audio capture concepts using self-contained implementations and mocks.

Covers:
- Ring buffer behaviour (write, read, overwrite on overflow)
- Subscribe / unsubscribe callback pattern
- get_buffer returns correct duration of audio
- Test signal generators (sine, noise)
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal ring buffer implementation for testing
# (Mirrors what would exist in an audio capture module)
# ---------------------------------------------------------------------------

class _RingBuffer:
    """A simple numpy ring buffer for audio capture testing."""

    def __init__(self, max_frames: int, dtype=np.float32):
        self._buf = np.zeros(max_frames, dtype=dtype)
        self._write_pos = 0
        self._length = 0
        self._max = max_frames

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n >= self._max:
            # Data larger than buffer — keep only the last max frames
            self._buf[:] = data[-self._max:]
            self._write_pos = 0
            self._length = self._max
            return
        end = self._write_pos + n
        if end <= self._max:
            self._buf[self._write_pos:end] = data
        else:
            first = self._max - self._write_pos
            self._buf[self._write_pos:] = data[:first]
            self._buf[:n - first] = data[first:]
        self._write_pos = end % self._max
        self._length = min(self._length + n, self._max)

    def read(self, num_frames: int = 0) -> np.ndarray:
        if num_frames <= 0 or num_frames > self._length:
            num_frames = self._length
        start = (self._write_pos - num_frames) % self._max
        if start + num_frames <= self._max:
            return self._buf[start:start + num_frames].copy()
        first = self._max - start
        return np.concatenate([self._buf[start:], self._buf[:num_frames - first]])

    @property
    def length(self) -> int:
        return self._length

    @property
    def capacity(self) -> int:
        return self._max


class _AudioCaptureMock:
    """Mock audio capture with ring buffer, subscriptions, and test generators."""

    def __init__(self, sample_rate: int = 48000, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self._buffer = _RingBuffer(int(sample_rate * buffer_seconds))
        self._subscribers = {}
        self._next_id = 0

    def write(self, samples: np.ndarray) -> None:
        self._buffer.write(samples)
        for callback in self._subscribers.values():
            callback(samples)

    def subscribe(self, callback) -> int:
        sub_id = self._next_id
        self._next_id += 1
        self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id: int) -> bool:
        return self._subscribers.pop(sub_id, None) is not None

    def get_buffer(self, duration_sec: float = 0.0) -> np.ndarray:
        if duration_sec <= 0:
            return self._buffer.read()
        frames = int(self.sample_rate * duration_sec)
        return self._buffer.read(frames)

    @staticmethod
    def generate_sine(freq_hz: float, duration_sec: float,
                      sample_rate: int = 48000, amplitude: float = 0.5):
        t = np.arange(int(sample_rate * duration_sec), dtype=np.float32) / sample_rate
        return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    @staticmethod
    def generate_noise(duration_sec: float, sample_rate: int = 48000,
                       amplitude: float = 0.1):
        n = int(sample_rate * duration_sec)
        return (amplitude * np.random.randn(n)).astype(np.float32)


class TestRingBuffer:
    """Tests for audio ring buffer functionality."""

    def test_ring_buffer_write_read(self):
        """Writing data then reading should return the same data."""
        buf = _RingBuffer(1000)
        data = np.arange(100, dtype=np.float32)
        buf.write(data)

        result = buf.read(100)
        np.testing.assert_array_almost_equal(result, data)

    def test_ring_buffer_overflow_wraps(self):
        """Writing more data than capacity should keep only the newest data."""
        buf = _RingBuffer(100)

        # Write 150 samples — only last 100 should remain
        data = np.arange(150, dtype=np.float32)
        buf.write(data)

        assert buf.length == 100
        result = buf.read(100)
        np.testing.assert_array_almost_equal(result, data[-100:])

    def test_ring_buffer_multiple_writes(self):
        """Multiple sequential writes should accumulate correctly."""
        buf = _RingBuffer(200)
        chunk1 = np.ones(80, dtype=np.float32) * 1.0
        chunk2 = np.ones(80, dtype=np.float32) * 2.0

        buf.write(chunk1)
        buf.write(chunk2)

        assert buf.length == 160
        result = buf.read(160)
        np.testing.assert_array_almost_equal(result[:80], chunk1)
        np.testing.assert_array_almost_equal(result[80:], chunk2)

    def test_ring_buffer_empty_read(self):
        """Reading from an empty buffer should return empty array."""
        buf = _RingBuffer(100)
        result = buf.read()
        assert len(result) == 0

    def test_ring_buffer_wrap_around(self):
        """Buffer should handle wrap-around correctly across multiple writes."""
        buf = _RingBuffer(100)

        # Fill to near capacity, then write more to force wrap
        buf.write(np.ones(90, dtype=np.float32) * 1.0)
        buf.write(np.ones(30, dtype=np.float32) * 2.0)

        assert buf.length == 100
        result = buf.read(100)
        # Last 100 samples: 70 ones followed by 30 twos
        np.testing.assert_array_almost_equal(result[:70], np.ones(70) * 1.0)
        np.testing.assert_array_almost_equal(result[70:], np.ones(30) * 2.0)


class TestSubscribeUnsubscribe:
    """Tests for audio subscriber pattern."""

    def test_subscribe_unsubscribe(self, sample_rate):
        """Subscribers should receive audio data and unsubscribe cleanly."""
        capture = _AudioCaptureMock(sample_rate=sample_rate)
        received = []

        sub_id = capture.subscribe(lambda data: received.append(data.copy()))

        # Write some audio
        tone = np.ones(480, dtype=np.float32) * 0.5
        capture.write(tone)

        assert len(received) == 1
        np.testing.assert_array_almost_equal(received[0], tone)

        # Unsubscribe
        result = capture.unsubscribe(sub_id)
        assert result is True

        # Write more — should not be received
        capture.write(tone)
        assert len(received) == 1  # Still 1

    def test_unsubscribe_invalid_id(self, sample_rate):
        """Unsubscribing with invalid ID should return False."""
        capture = _AudioCaptureMock(sample_rate=sample_rate)
        assert capture.unsubscribe(999) is False

    def test_multiple_subscribers(self, sample_rate):
        """Multiple subscribers should all receive data."""
        capture = _AudioCaptureMock(sample_rate=sample_rate)
        counts = [0, 0]

        capture.subscribe(lambda data: counts.__setitem__(0, counts[0] + 1))
        capture.subscribe(lambda data: counts.__setitem__(1, counts[1] + 1))

        capture.write(np.zeros(100, dtype=np.float32))
        assert counts == [1, 1]


class TestGetBufferDuration:
    """Tests for get_buffer returning correct duration of audio."""

    def test_get_buffer_returns_correct_duration(self, sample_rate):
        """get_buffer(duration_sec) should return the correct number of frames."""
        capture = _AudioCaptureMock(sample_rate=sample_rate, buffer_seconds=5.0)

        # Write 2 seconds of audio
        data = np.random.randn(sample_rate * 2).astype(np.float32)
        capture.write(data)

        # Request 1 second
        buf = capture.get_buffer(duration_sec=1.0)
        expected_frames = sample_rate  # 1 second
        assert len(buf) == expected_frames, (
            f"Expected {expected_frames} frames, got {len(buf)}"
        )

    def test_get_buffer_all(self, sample_rate):
        """get_buffer with no duration should return all available data."""
        capture = _AudioCaptureMock(sample_rate=sample_rate, buffer_seconds=5.0)

        n = sample_rate  # 1 second
        data = np.random.randn(n).astype(np.float32)
        capture.write(data)

        buf = capture.get_buffer()
        assert len(buf) == n

    def test_get_buffer_more_than_available(self, sample_rate):
        """Requesting more than available should return what is available."""
        capture = _AudioCaptureMock(sample_rate=sample_rate, buffer_seconds=5.0)

        # Write 0.5 seconds
        data = np.random.randn(sample_rate // 2).astype(np.float32)
        capture.write(data)

        # Request 2 seconds
        buf = capture.get_buffer(duration_sec=2.0)
        assert len(buf) == sample_rate // 2


class TestTestGenerators:
    """Tests for audio test signal generators."""

    def test_test_generators_sine(self, sample_rate):
        """generate_sine should produce a sine wave of correct length and frequency."""
        tone = _AudioCaptureMock.generate_sine(
            freq_hz=1000.0, duration_sec=1.0,
            sample_rate=sample_rate, amplitude=0.5,
        )

        assert len(tone) == sample_rate
        assert tone.dtype == np.float32
        assert np.max(np.abs(tone)) <= 0.51  # Close to amplitude

    def test_test_generators_noise(self, sample_rate):
        """generate_noise should produce noise of correct length."""
        noise = _AudioCaptureMock.generate_noise(
            duration_sec=1.0, sample_rate=sample_rate, amplitude=0.1,
        )

        assert len(noise) == sample_rate
        assert noise.dtype == np.float32
        # Standard deviation should be close to amplitude
        assert abs(np.std(noise) - 0.1) < 0.05

    def test_sine_frequency_content(self, sample_rate):
        """Generated sine should have energy at the expected frequency."""
        tone = _AudioCaptureMock.generate_sine(
            freq_hz=440.0, duration_sec=0.5, sample_rate=sample_rate,
        )

        # FFT to find dominant frequency
        spectrum = np.abs(np.fft.rfft(tone))
        freqs = np.fft.rfftfreq(len(tone), 1.0 / sample_rate)
        peak_freq = freqs[np.argmax(spectrum)]

        assert abs(peak_freq - 440.0) < 5.0, (
            f"Dominant frequency should be ~440 Hz, got {peak_freq}"
        )
