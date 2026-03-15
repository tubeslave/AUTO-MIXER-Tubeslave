"""Tests for audio_capture module."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pytest
from audio_capture import AudioCapture


class TestAudioCapture:
    def test_init(self):
        capture = AudioCapture(num_channels=8, sample_rate=48000)
        assert capture is not None
        assert capture.num_channels == 8
        assert capture.sample_rate == 48000

    def test_test_mode(self):
        """Test that test signal generation works."""
        capture = AudioCapture(num_channels=2, sample_rate=48000)
        assert capture.source_type is not None
