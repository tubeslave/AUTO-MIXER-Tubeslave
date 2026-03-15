"""Shared test fixtures for AUTO-MIXER-Tubeslave."""
import sys
import os
import pytest
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


@pytest.fixture
def sample_rate():
    return 48000


@pytest.fixture
def block_size():
    return 1024


@pytest.fixture
def sine_wave(sample_rate):
    """Generate a 1kHz sine wave, 1 second."""
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    return np.sin(2 * np.pi * 1000 * t) * 0.5


@pytest.fixture
def pink_noise(sample_rate):
    """Generate pink noise, 1 second."""
    n = sample_rate
    white = np.random.randn(n).astype(np.float32)
    # Simple pink noise approximation via cumulative filter
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510])
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])
    try:
        from scipy.signal import lfilter
        pink = lfilter(b, a, white)
    except ImportError:
        pink = np.convolve(white, [0.5, 0.3, 0.2], mode='same')
    return (pink / (np.max(np.abs(pink)) + 1e-10) * 0.5).astype(np.float32)


@pytest.fixture
def silence(sample_rate):
    """Generate silence, 1 second."""
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture
def stereo_audio(sine_wave):
    """Generate stereo audio."""
    return np.stack([sine_wave, sine_wave * 0.8])


@pytest.fixture
def multi_channel_audio(sample_rate):
    """Generate 8-channel audio for testing."""
    channels = []
    for i in range(8):
        freq = 100 * (i + 1)
        t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
        channels.append(np.sin(2 * np.pi * freq * t) * 0.3)
    return channels
