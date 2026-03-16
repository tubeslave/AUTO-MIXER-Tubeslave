"""
Tests for audio device discovery functions and SoundGrid/Dante detection.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from audio_capture import (
    AudioSourceType, AudioDeviceType,
    list_audio_devices, find_device_by_name, detect_audio_device,
    SOUNDGRID_PATTERNS,
)


class TestAudioSourceType:
    def test_soundgrid_exists(self):
        assert AudioSourceType.SOUNDGRID.value == "soundgrid"

    def test_dante_exists(self):
        assert AudioSourceType.DANTE.value == "dante"

    def test_silence_exists(self):
        assert AudioSourceType.SILENCE.value == "silence"


class TestAudioDeviceType:
    def test_soundgrid(self):
        assert AudioDeviceType.SOUNDGRID.value == "soundgrid"

    def test_dante(self):
        assert AudioDeviceType.DANTE.value == "dante"

    def test_default(self):
        assert AudioDeviceType.DEFAULT.value == "default"


MOCK_DEVICES = [
    {"name": "Waves SoundGrid Driver", "max_input_channels": 64, "default_samplerate": 48000.0, "hostapi": 0},
    {"name": "Dante Virtual Soundcard", "max_input_channels": 64, "default_samplerate": 48000.0, "hostapi": 0},
    {"name": "Built-in Microphone", "max_input_channels": 2, "default_samplerate": 44100.0, "hostapi": 0},
    {"name": "HDMI Output", "max_input_channels": 0, "default_samplerate": 48000.0, "hostapi": 0},
]


def _mock_query_devices():
    return MOCK_DEVICES


class TestListAudioDevices:
    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_lists_input_devices_only(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        devices = list_audio_devices()
        # HDMI Output has 0 input channels, should be excluded
        assert len(devices) == 3
        names = [d["name"] for d in devices]
        assert "HDMI Output" not in names

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_includes_device_index(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        devices = list_audio_devices()
        indices = [d["index"] for d in devices]
        assert 0 in indices  # SoundGrid
        assert 1 in indices  # Dante

    @patch("audio_capture.HAS_SOUNDDEVICE", False)
    def test_returns_empty_without_sounddevice(self):
        devices = list_audio_devices()
        assert devices == []


class TestFindDeviceByName:
    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_finds_soundgrid(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        idx = find_device_by_name("waves")
        assert idx == 0

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_finds_dante(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        idx = find_device_by_name("dante")
        assert idx == 1

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_case_insensitive(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        idx = find_device_by_name("WAVES")
        assert idx == 0

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_returns_none_for_unknown(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        idx = find_device_by_name("nonexistent")
        assert idx is None


class TestDetectAudioDevice:
    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_prefers_soundgrid(self, mock_sd):
        mock_sd.query_devices.return_value = MOCK_DEVICES
        idx, dtype = detect_audio_device()
        assert dtype == AudioDeviceType.SOUNDGRID
        assert idx == 0

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_falls_back_to_dante(self, mock_sd):
        # Remove SoundGrid device
        devices = [d for d in MOCK_DEVICES if "waves" not in d["name"].lower() and "soundgrid" not in d["name"].lower()]
        mock_sd.query_devices.return_value = devices
        idx, dtype = detect_audio_device()
        assert dtype == AudioDeviceType.DANTE

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_falls_back_to_default(self, mock_sd):
        # Only generic devices
        devices = [{"name": "Built-in Mic", "max_input_channels": 2, "default_samplerate": 44100.0, "hostapi": 0}]
        mock_sd.query_devices.return_value = devices
        idx, dtype = detect_audio_device()
        assert dtype == AudioDeviceType.DEFAULT
        assert idx is None

    @patch("audio_capture.HAS_SOUNDDEVICE", True)
    @patch("audio_capture.sd", create=True)
    def test_sg_pattern_match(self, mock_sd):
        devices = [{"name": "SG Driver v2", "max_input_channels": 32, "default_samplerate": 48000.0, "hostapi": 0}]
        mock_sd.query_devices.return_value = devices
        idx, dtype = detect_audio_device()
        assert dtype == AudioDeviceType.SOUNDGRID


class TestSoundGridPatterns:
    def test_patterns_are_lowercase(self):
        for p in SOUNDGRID_PATTERNS:
            assert p == p.lower()
