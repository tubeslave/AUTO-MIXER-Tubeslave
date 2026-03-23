"""
Tests for DLiveClient MIDI message building, value conversions,
channel selection, and mixer factory function.
"""

import sys
import os
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dlive_client import (
    db_to_fader_value, fader_value_to_db,
    db_to_gain_value, freq_to_nrpn, db_to_eq_gain, q_to_nrpn,
    build_nrpn, build_note_on, build_pitchbend, build_sysex, build_program_change,
    FADER_MAX, FADER_0DB, FADER_NEG_INF, SYSEX_HEADER, SYSEX_END,
    CHANNEL_TYPE_OFFSET, DLiveClient,
)
from mixer_client_base import MixerClientBase, create_mixer_client


class TestFaderConversions:
    """Test dB ↔ fader value conversions."""

    def test_neg_inf_to_fader(self):
        assert db_to_fader_value(-100.0) == FADER_NEG_INF
        assert db_to_fader_value(-200.0) == FADER_NEG_INF

    def test_0db_to_fader(self):
        assert db_to_fader_value(0.0) == FADER_0DB

    def test_plus10_to_fader(self):
        assert db_to_fader_value(10.0) == FADER_MAX

    def test_above_max_clamps(self):
        assert db_to_fader_value(20.0) == FADER_MAX

    def test_fader_to_db_zero(self):
        assert fader_value_to_db(0) == -100.0

    def test_fader_to_db_0db(self):
        result = fader_value_to_db(FADER_0DB)
        assert abs(result) < 0.1  # Should be ~0 dB

    def test_fader_to_db_max(self):
        result = fader_value_to_db(FADER_MAX)
        assert abs(result - 10.0) < 0.1

    def test_roundtrip_0db(self):
        val = db_to_fader_value(0.0)
        db = fader_value_to_db(val)
        assert abs(db) < 0.1

    def test_roundtrip_minus50(self):
        val = db_to_fader_value(-50.0)
        db = fader_value_to_db(val)
        assert abs(db - (-50.0)) < 1.0  # Allow some quantization error


class TestGainConversion:
    def test_neg_inf(self):
        assert db_to_gain_value(-100.0) == 0

    def test_max_gain(self):
        assert db_to_gain_value(60.0) == FADER_MAX

    def test_mid_range(self):
        val = db_to_gain_value(0.0)
        assert 0 < val < FADER_MAX


class TestEQConversions:
    def test_freq_to_nrpn_20hz(self):
        val = freq_to_nrpn(20.0)
        assert val == 0

    def test_freq_to_nrpn_20khz(self):
        val = freq_to_nrpn(20000.0)
        assert val == FADER_MAX

    def test_eq_gain_zero(self):
        val = db_to_eq_gain(0.0)
        # 0 dB should map to midpoint
        assert abs(val - FADER_MAX // 2) < 100

    def test_eq_gain_clamps(self):
        assert db_to_eq_gain(-20.0) == 0
        assert db_to_eq_gain(20.0) == FADER_MAX

    def test_q_to_nrpn_min(self):
        val = q_to_nrpn(0.3)
        assert val == 0

    def test_q_to_nrpn_max(self):
        val = q_to_nrpn(35.0)
        assert val == FADER_MAX


class TestMIDIBuilders:
    """Test MIDI message building (pure functions)."""

    def test_build_nrpn_length(self):
        msg = build_nrpn(0, 0x40, 0x00, 8192)
        assert len(msg) == 12  # 4 CC messages × 3 bytes

    def test_build_nrpn_channel(self):
        msg = build_nrpn(5, 0x40, 0x00, 0)
        # First byte should be 0xB5
        assert msg[0] == 0xB5

    def test_build_nrpn_msb_lsb(self):
        msg = build_nrpn(0, 0x40, 0x20, 0)
        assert msg[1] == 0x63  # NRPN MSB CC
        assert msg[2] == 0x40  # MSB value
        assert msg[4] == 0x62  # NRPN LSB CC
        assert msg[5] == 0x20  # LSB value

    def test_build_note_on(self):
        msg = build_note_on(0, 60, 127)
        assert len(msg) == 3
        assert msg[0] == 0x90
        assert msg[1] == 60
        assert msg[2] == 127

    def test_build_note_on_channel(self):
        msg = build_note_on(4, 0, 0)
        assert msg[0] == 0x94

    def test_build_pitchbend(self):
        msg = build_pitchbend(0, 8192)
        assert len(msg) == 3
        assert msg[0] == 0xE0
        # 8192 = 0x2000 → LSB=0, MSB=0x40
        assert msg[1] == 0x00  # LSB
        assert msg[2] == 0x40  # MSB

    def test_build_sysex(self):
        payload = bytes([0x01, 0x02])
        msg = build_sysex(payload)
        assert msg[:len(SYSEX_HEADER)] == SYSEX_HEADER
        assert msg[-1] == SYSEX_END
        assert msg[len(SYSEX_HEADER):-1] == payload

    def test_build_program_change(self):
        msg = build_program_change(0, 0, 1, 25)
        assert len(msg) == 8
        # Bank Select MSB
        assert msg[0] == 0xB0
        assert msg[1] == 0x00
        assert msg[2] == 0x00
        # Bank Select LSB
        assert msg[3] == 0xB0
        assert msg[4] == 0x20
        assert msg[5] == 0x01
        # Program Change
        assert msg[6] == 0xC0
        assert msg[7] == 25


class TestDLiveClientChannelSelection:
    """Test MIDI channel selection for different channel types."""

    def test_input_channel(self):
        client = DLiveClient.__new__(DLiveClient)
        client.midi_base_channel = 0
        assert client._midi_channel("input") == 0

    def test_dca_channel(self):
        client = DLiveClient.__new__(DLiveClient)
        client.midi_base_channel = 0
        assert client._midi_channel("dca") == 4

    def test_base_channel_offset(self):
        client = DLiveClient.__new__(DLiveClient)
        client.midi_base_channel = 2
        assert client._midi_channel("input") == 2
        assert client._midi_channel("dca") == 6

    def test_channel_wraps_at_16(self):
        client = DLiveClient.__new__(DLiveClient)
        client.midi_base_channel = 14
        # 14 + 4 = 18, should wrap to 2 (& 0x0F)
        assert client._midi_channel("dca") == (18 & 0x0F)


class TestDLiveClientIsABC:
    """Verify DLiveClient inherits from MixerClientBase."""

    def test_is_subclass(self):
        assert issubclass(DLiveClient, MixerClientBase)


class TestMixerFactory:
    """Test create_mixer_client factory function."""

    def test_dlive_creation(self):
        client = create_mixer_client("dlive", {"ip": "10.0.0.1", "port": 51329, "tls": True})
        assert isinstance(client, DLiveClient)
        assert client.ip == "10.0.0.1"
        assert client.port == 51329
        assert client.tls is True

    def test_unknown_mixer_raises(self):
        with pytest.raises(ValueError, match="Unknown mixer type"):
            create_mixer_client("yamaha", {})

    def test_case_insensitive(self):
        client = create_mixer_client("DLive", {})
        assert isinstance(client, DLiveClient)
