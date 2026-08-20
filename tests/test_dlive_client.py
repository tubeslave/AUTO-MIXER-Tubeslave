"""
Tests for DLiveClient MIDI message building, value conversions,
channel selection, and mixer factory function.
"""

import sys
import os
import ssl
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dlive_client import (
    db_to_fader_value, fader_value_to_db,
    db_to_gain_value, freq_to_nrpn, db_to_eq_gain, q_to_nrpn,
    build_nrpn, build_note_on, build_pitchbend, build_sysex, build_program_change,
    build_get_fader, build_get_mute, build_get_name,
    validate_dlive_connection_config,
    FADER_MAX, FADER_0DB, FADER_NEG_INF, SYSEX_HEADER, SYSEX_END,
    DLIVE_MAX_SYSEX_FRAME_BYTES,
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
        val = db_to_gain_value(32.5)
        assert 0 < val < FADER_MAX


class TestEQConversions:
    def test_freq_to_nrpn_20hz(self):
        val = freq_to_nrpn(20.0)
        assert val == 0

    def test_freq_to_nrpn_20khz(self):
        val = freq_to_nrpn(20000.0)
        assert val == FADER_MAX

    def test_eq_gain_zero(self):
        assert db_to_eq_gain(0.0) == 63

    def test_eq_gain_clamps(self):
        assert db_to_eq_gain(-20.0) == 0
        assert db_to_eq_gain(20.0) == 126

    def test_q_to_nrpn_min(self):
        val = q_to_nrpn(0.3)
        assert val == 0

    def test_q_to_nrpn_max(self):
        val = q_to_nrpn(35.0)
        assert val == 24


class TestMIDIBuilders:
    """Test MIDI message building (pure functions)."""

    def test_build_nrpn_length(self):
        msg = build_nrpn(0, 0x04, 0x17, 0x61)
        assert msg == bytes.fromhex("B0 63 04 62 17 06 61")

    def test_build_nrpn_channel(self):
        msg = build_nrpn(5, 0x40, 0x17, 0)
        # First byte should be 0xB5
        assert msg[0] == 0xB5
        assert msg.count(0xB5) == 1

    def test_build_nrpn_msb_lsb(self):
        msg = build_nrpn(0, 0x40, 0x20, 0)
        assert msg[1] == 0x63  # NRPN MSB CC
        assert msg[2] == 0x40  # MSB value
        assert msg[3] == 0x62  # NRPN LSB CC under running status
        assert msg[4] == 0x20  # LSB value
        assert msg[5] == 0x06  # Data Entry CC under running status

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
        assert build_pitchbend(0, 0x20, 0x40) == bytes.fromhex("E0 20 40")

    def test_build_sysex(self):
        payload = bytes([0x01, 0x02])
        msg = build_sysex(payload)
        assert msg[:len(SYSEX_HEADER)] == SYSEX_HEADER
        assert msg[-1] == SYSEX_END
        assert msg[len(SYSEX_HEADER):-1] == payload

    def test_build_program_change(self):
        assert build_program_change(0, 1, 25) == bytes.fromhex("B0 00 01 C0 19")

    def test_read_queries_match_official_vectors(self):
        assert build_get_fader(0, 4) == bytes.fromhex(
            "F0 00 00 1A 50 10 01 00 00 05 0B 17 04 F7"
        )
        assert build_get_mute(0, 4) == bytes.fromhex(
            "F0 00 00 1A 50 10 01 00 00 05 09 04 F7"
        )
        assert build_get_name(0, 4) == bytes.fromhex(
            "F0 00 00 1A 50 10 01 00 00 01 04 F7"
        )


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

    def test_base_channel_cannot_overlap_past_midi_16(self):
        with pytest.raises(ValueError, match="out_of_range_0_11"):
            DLiveClient(midi_base_channel=12)


class TestConnectionPreflight:
    def test_surface_and_mixrack_ports_are_distinct(self):
        assert validate_dlive_connection_config(host="192.168.3.70")["port"] == 51328
        report = validate_dlive_connection_config(
            host="192.168.3.70", endpoint="mixrack", tls=False
        )
        assert report["port"] == 51325
        assert report["console_contacted"] is False
        assert report["live_authority_granted"] is False

    def test_port_endpoint_mismatch_fails_before_socket(self):
        with pytest.raises(ValueError, match="expected_51325"):
            DLiveClient(ip="192.168.3.70", endpoint="mixrack", port=51328)

    @pytest.mark.parametrize("host", ["", "bad host", "300.1.1.1"])
    def test_invalid_host_rejected(self, host):
        with pytest.raises(ValueError, match="host_invalid"):
            validate_dlive_connection_config(host=host)


class TestRunningStatus:
    def test_parser_accepts_official_running_status_nrpn(self):
        client = DLiveClient()
        client._parse_incoming(bytes.fromhex("B0 63 04 62 17 06 61"))
        assert client.state["fader/5"] == 0x61

    def test_running_status_survives_tcp_split(self):
        message = bytes.fromhex("B0 63 04 62 17 06 61")
        for split in range(1, len(message)):
            client = DLiveClient()
            client._parse_incoming(message[:split])
            client._parse_incoming(message[split:])
            assert client.state["fader/5"] == 0x61


class TestSysExResourceBudget:
    def test_unterminated_sysex_cannot_grow_parse_buffer_without_bound(self):
        client = DLiveClient()
        resource_budget = DLIVE_MAX_SYSEX_FRAME_BYTES

        client._parse_incoming(b"\xF0" + (b"\x01" * resource_budget))

        assert len(client._parse_buffer) == 0
        assert client._discarding_oversized_sysex is True

        for _ in range(8):
            client._parse_incoming(b"\x01" * 4096)
            assert len(client._parse_buffer) == 0

    def test_oversized_sysex_is_discarded_until_terminator_then_parser_recovers(self):
        client = DLiveClient()
        resource_budget = DLIVE_MAX_SYSEX_FRAME_BYTES
        embedded_note = build_note_on(0, 5, 0x7F)
        legitimate_note = build_note_on(0, 4, 0x7F)
        client._parse_incoming(
            b"\xF0"
            + (b"\x01" * resource_budget)
            + embedded_note
            + bytes((SYSEX_END,))
            + legitimate_note
        )

        assert "note/0/5" not in client.state
        assert client.state["note/0/4"] == 0x7F
        assert client._discarding_oversized_sysex is False

    def test_fragmented_sysex_within_budget_remains_supported(self):
        client = DLiveClient()
        message = build_sysex(bytes((0x00, 0x02, 0x04)) + b"Lead")

        for byte in message:
            client._parse_incoming(bytes((byte,)))

        assert client.state["name/5"] == "Lead"

    def test_new_connection_resets_oversized_sysex_discard_state(self, monkeypatch):
        client = DLiveClient(ip="127.0.0.1")
        client._parse_incoming(
            b"\xF0" + (b"\x01" * DLIVE_MAX_SYSEX_FRAME_BYTES)
        )
        assert client._discarding_oversized_sysex is True

        raw = _FakeSocket()
        monkeypatch.setattr("dlive_client.socket.socket", lambda *args, **kwargs: raw)
        monkeypatch.setattr("dlive_client.threading.Thread", _NoopThread)

        assert client.connect() is True
        assert client._discarding_oversized_sysex is False
        assert len(client._parse_buffer) == 0

        client._parse_incoming(build_note_on(0, 4, 0x7F))
        assert client.state["note/0/4"] == 0x7F
        client.disconnect()


class TestDLiveClientIsABC:
    """Verify DLiveClient inherits from MixerClientBase."""

    def test_is_subclass(self):
        assert issubclass(DLiveClient, MixerClientBase)

    def test_private_fader_hook_rejects_duck_typed_fake_authority(self):
        client = DLiveClient()
        sent = []
        client._send_raw = lambda payload: sent.append(bytes(payload)) or True

        class FakeAuthority:
            def consume_fader_write(self, **kwargs):
                return True

        assert client._supervised_fader_write(
            1,
            -6.0,
            authority=FakeAuthority(),
            phase="apply",
        ) is False
        assert sent == []


class _FakeSocket:
    def __init__(self):
        self.connected_to = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def setsockopt(self, *args):
        self.sockopt = args

    def connect(self, address):
        self.connected_to = address

    def sendall(self, data):
        self.sent = data

    def recv(self, size):
        return b"AuthOK"

    def close(self):
        self.closed = True


class _FakeSSLContext:
    def __init__(self, cafile=None):
        self.cafile = cafile
        self.wrapped = []

    def wrap_socket(self, raw, *, server_hostname):
        self.wrapped.append((raw, server_hostname))
        return raw


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class TestDLiveClientTLS:
    def test_tls_uses_default_certificate_verification(self, monkeypatch):
        raw = _FakeSocket()
        contexts = []

        def fake_socket(*args, **kwargs):
            return raw

        def fake_create_default_context(*, cafile=None):
            context = _FakeSSLContext(cafile=cafile)
            contexts.append(context)
            return context

        monkeypatch.setattr("dlive_client.socket.socket", fake_socket)
        monkeypatch.setattr(
            "dlive_client.ssl.create_default_context",
            fake_create_default_context,
        )
        monkeypatch.setattr("dlive_client.threading.Thread", _NoopThread)

        client = DLiveClient(
            ip="10.0.0.1",
            port=51329,
            tls=True,
            tls_ca_file="/secure/dlive-ca.pem",
            tls_server_hostname="dlive-console.local",
            password="test-password",
        )

        assert client.connect() is True
        assert raw.connected_to == ("10.0.0.1", 51329)
        assert contexts[0].cafile == "/secure/dlive-ca.pem"
        assert contexts[0].wrapped == [(raw, "dlive-console.local")]
        assert raw.sent == b"\x00,test-password"

    def test_tls_connect_fails_closed_on_certificate_error(self, monkeypatch):
        raw = _FakeSocket()

        def fake_socket(*args, **kwargs):
            return raw

        class RejectingSSLContext:
            def wrap_socket(self, raw_sock, *, server_hostname):
                raise ssl.SSLCertVerificationError("certificate verify failed")

        monkeypatch.setattr("dlive_client.socket.socket", fake_socket)
        monkeypatch.setattr(
            "dlive_client.ssl.create_default_context",
            lambda *, cafile=None: RejectingSSLContext(),
        )

        client = DLiveClient(
            ip="10.0.0.1",
            port=51329,
            tls=True,
            tls_ca_file="/secure/dlive-ca.pem",
            tls_server_hostname="dlive-console.local",
            password="test-password",
        )

        assert client.connect() is False
        assert client.is_connected is False

    def test_non_loopback_cleartext_fails_before_socket(self, monkeypatch):
        socket_factory = pytest.fail
        monkeypatch.setattr("dlive_client.socket.socket", socket_factory)

        client = DLiveClient(ip="10.0.0.1", port=51328, tls=False)

        assert client.connect() is False
        assert client.is_connected is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "tls_ca_file": None,
                "tls_server_hostname": "dlive-console.local",
                "password": "test-password",
            },
            {
                "tls_ca_file": "/secure/dlive-ca.pem",
                "tls_server_hostname": None,
                "password": "test-password",
            },
            {
                "tls_ca_file": "/secure/dlive-ca.pem",
                "tls_server_hostname": "dlive-console.local",
                "password": "",
            },
        ],
    )
    def test_tls_identity_and_password_are_required_before_socket(
        self,
        monkeypatch,
        kwargs,
    ):
        monkeypatch.setattr("dlive_client.socket.socket", pytest.fail)
        client = DLiveClient(ip="10.0.0.1", port=51329, tls=True, **kwargs)

        assert client.connect() is False
        assert client.is_connected is False


class TestMixerFactory:
    """Test create_mixer_client factory function."""

    def test_dlive_creation(self):
        client = create_mixer_client(
            "dlive",
            {
                "ip": "10.0.0.1",
                "port": 51329,
                "tls": True,
                "endpoint": "surface",
                "dliveTlsCaFile": "/secure/dlive-ca.pem",
                "dliveTlsServerHostname": "dlive-console.local",
            },
        )
        assert isinstance(client, DLiveClient)
        assert client.ip == "10.0.0.1"
        assert client.port == 51329
        assert client.tls is True
        assert client.endpoint == "surface"
        assert client.tls_ca_file == "/secure/dlive-ca.pem"
        assert client.tls_server_hostname == "dlive-console.local"

    def test_unknown_mixer_raises(self):
        with pytest.raises(ValueError, match="Unknown mixer type"):
            create_mixer_client("yamaha", {})

    def test_case_insensitive(self):
        client = create_mixer_client("DLive", {})
        assert isinstance(client, DLiveClient)
