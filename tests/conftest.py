"""
Shared pytest fixtures for AUTO MIXER Tubeslave test suite.

Provides audio test signals, mock OSC client, temp directories,
and ensures backend modules are importable.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: make backend/ importable without installation
# ---------------------------------------------------------------------------

_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
_BACKEND_DIR = os.path.abspath(_BACKEND_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_rate():
    """Standard sample rate used across all tests."""
    return 48000


@pytest.fixture
def test_audio_sine(sample_rate):
    """
    Generate a 1 kHz sine wave at -6 dBFS, 1 second long.

    Returns:
        np.ndarray of float32 samples.
    """
    duration = 1.0
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    amplitude = 10 ** (-6.0 / 20.0)  # -6 dBFS
    signal = (amplitude * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    return signal


@pytest.fixture
def test_audio_pink_noise(sample_rate):
    """
    Generate approximately pink noise (1/f spectrum) using the Voss-McCartney
    algorithm, 1 second long, normalised to -12 dBFS.

    Returns:
        np.ndarray of float32 samples.
    """
    rng = np.random.RandomState(42)
    n_samples = int(sample_rate * 1.0)
    n_rows = 16
    array = rng.randn(n_rows, n_samples)
    # Cumulative sum along rows gives approximate 1/f spectrum
    pink = np.sum(array, axis=0)
    # Normalise to -12 dBFS
    peak = np.max(np.abs(pink))
    if peak > 0:
        target_amp = 10 ** (-12.0 / 20.0)
        pink = pink / peak * target_amp
    return pink.astype(np.float32)


@pytest.fixture
def test_audio_silence(sample_rate):
    """
    Generate 1 second of digital silence.

    Returns:
        np.ndarray of float32 zeros.
    """
    return np.zeros(sample_rate, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock OSC / Wing client
# ---------------------------------------------------------------------------

class _MockOSCClient:
    """Lightweight mock that records OSC sends for assertion."""

    def __init__(self):
        self.sent_messages = []
        self.state = {}
        self.is_connected = True
        self._osc_throttle_enabled = False
        self._osc_throttle_hz = 10.0

    def send(self, address, *values):
        self.sent_messages.append((address, values))
        if values:
            self.state[address] = values[-1]
        return True

    def set_osc_throttle(self, enabled=True, hz=10.0):
        self._osc_throttle_enabled = enabled
        self._osc_throttle_hz = hz

    def subscribe(self, address_pattern, callback):
        pass

    def connect(self, timeout=5.0):
        self.is_connected = True
        return True

    def disconnect(self):
        self.is_connected = False

    def reset(self):
        self.sent_messages.clear()
        self.state.clear()


@pytest.fixture
def mock_osc_client():
    """Provide a mock OSC client that records all sent messages."""
    return _MockOSCClient()


# ---------------------------------------------------------------------------
# Temporary directory
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="tubeslave_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
