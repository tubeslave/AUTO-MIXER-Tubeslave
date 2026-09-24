import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.wing_main_meter import (
    CONTROL_CHANNEL_SELECT,
    DEFAULT_REPORT_ID,
    WingMainMeterReadError,
    WingNativeMainMeterProvider,
    build_main_meter_request,
    decode_main_meter_datagram,
)


class FakeUdpSocket:
    def __init__(self, datagrams, *, port=14135, receive_error=None):
        self.datagrams = list(datagrams)
        self.port = port
        self.receive_error = receive_error
        self.bound = None
        self.timeout = None
        self.closed = False

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return ('0.0.0.0', self.port)

    def settimeout(self, timeout):
        self.timeout = timeout

    def recvfrom(self, size):
        if self.receive_error is not None:
            raise self.receive_error
        if not self.datagrams:
            raise TimeoutError('no datagram')
        return self.datagrams.pop(0), ('192.0.2.10', 2222)

    def close(self):
        self.closed = True


class FakeTcpSocket:
    def __init__(self):
        self.timeout = None
        self.sent = []
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendall(self, data):
        self.sent.append(bytes(data))

    def close(self):
        self.closed = True


def _meter_datagram(
    *,
    report_id=DEFAULT_REPORT_ID,
    input_left=-10.0,
    input_right=-11.0,
    output_left=-6.0,
    output_right=-7.0,
):
    def word(db):
        return int(round(db * 256.0))

    words = (
        word(input_left),
        word(input_right),
        word(output_left),
        word(output_right),
        word(-2.0),
        -100,
        word(-3.0),
        -200,
    )
    return struct.pack('>I8h', report_id, *words)


def test_main_meter_request_matches_documented_native_sequence_for_main_1():
    request = build_main_meter_request(
        udp_port=0x3737,
        main=1,
        report_id=DEFAULT_REPORT_ID,
    )

    assert request == bytes.fromhex(
        'dfd3'      # select native meter-request channel
        'd33737'    # declare UDP return port 0x3737
        'd44d41494e'  # report id "MAIN"
        'dca300de'  # collection: MAIN 1 (wire index 0)
        'dfd1'      # return to ordinary control channel
    )


def test_main_meter_request_escapes_literal_df_inside_numeric_values():
    request = build_main_meter_request(
        udp_port=0xDF37,
        main=4,
        report_id=0x12DF34DF,
    )

    assert bytes.fromhex('d3dfde37') in request
    assert bytes.fromhex('d412dfde34dfde') in request
    assert bytes.fromhex('dca303de') in request


def test_main_meter_datagram_decodes_signed_big_endian_levels_in_1_over_256_db():
    frame = decode_main_meter_datagram(
        _meter_datagram(),
        expected_report_id=DEFAULT_REPORT_ID,
    )

    assert frame.input_left_dbfs == pytest.approx(-10.0)
    assert frame.input_right_dbfs == pytest.approx(-11.0)
    assert frame.output_left_dbfs == pytest.approx(-6.0)
    assert frame.output_right_dbfs == pytest.approx(-7.0)
    assert frame.output_level_dbfs == pytest.approx(-6.0)
    assert frame.gate_gain_raw == -100
    assert frame.dyn_gain_raw == -200


def test_main_meter_datagram_rejects_wrong_report_id_and_wrong_shape():
    with pytest.raises(ValueError, match='report id mismatch'):
        decode_main_meter_datagram(
            _meter_datagram(report_id=0x12345678),
            expected_report_id=DEFAULT_REPORT_ID,
        )

    with pytest.raises(ValueError, match='exactly 20 bytes'):
        decode_main_meter_datagram(b'bad', expected_report_id=DEFAULT_REPORT_ID)


def test_provider_reads_one_matching_main_frame_and_returns_independent_evidence():
    udp = FakeUdpSocket([_meter_datagram()], port=14135)
    tcp = FakeTcpSocket()
    connections = []
    times = iter((10.0, 10.0, 10.05))

    def connect(address, timeout):
        connections.append((address, timeout))
        return tcp

    provider = WingNativeMainMeterProvider(
        '192.0.2.10',
        main=1,
        timeout_s=0.35,
        time_fn=lambda: next(times),
        udp_socket_factory=lambda: udp,
        tcp_connection_factory=connect,
    )

    evidence = provider.read()

    assert connections == [(('192.0.2.10', 2222), 0.35)]
    assert udp.bound == ('0.0.0.0', 0)
    assert udp.timeout == pytest.approx(0.35)
    assert tcp.sent[0] == CONTROL_CHANNEL_SELECT
    assert tcp.sent[1] == build_main_meter_request(
        udp_port=14135,
        main=1,
        report_id=DEFAULT_REPORT_ID,
    )
    assert evidence.peak_dbfs == pytest.approx(-6.0)
    assert evidence.rms_dbfs is None
    assert evidence.timestamp_s == pytest.approx(10.05)
    assert evidence.source == 'wing-native-main-meter:MAIN.1'
    assert udp.closed is True
    assert tcp.closed is True


def test_provider_skips_wrong_report_id_then_accepts_matching_frame():
    udp = FakeUdpSocket(
        [
            _meter_datagram(report_id=0x01020304, output_left=-20.0),
            _meter_datagram(output_left=-8.0, output_right=-9.0),
        ]
    )
    tcp = FakeTcpSocket()
    times = iter((1.0, 1.0, 1.01, 1.02))
    provider = WingNativeMainMeterProvider(
        '192.0.2.10',
        time_fn=lambda: next(times),
        udp_socket_factory=lambda: udp,
        tcp_connection_factory=lambda address, timeout: tcp,
    )

    evidence = provider.read()

    assert evidence.peak_dbfs == pytest.approx(-8.0)
    assert evidence.timestamp_s == pytest.approx(1.02)
    assert udp.datagrams == []


def test_provider_timeout_is_fail_closed_and_sockets_are_closed():
    udp = FakeUdpSocket([], receive_error=TimeoutError('expired'))
    tcp = FakeTcpSocket()
    provider = WingNativeMainMeterProvider(
        '192.0.2.10',
        time_fn=lambda: 1.0,
        udp_socket_factory=lambda: udp,
        tcp_connection_factory=lambda address, timeout: tcp,
    )

    with pytest.raises(WingMainMeterReadError, match='timed out waiting'):
        provider.read()

    assert udp.closed is True
    assert tcp.closed is True


def test_provider_rejects_invalid_main_number_before_opening_transport():
    with pytest.raises(ValueError, match='main must be an integer in 1..4'):
        WingNativeMainMeterProvider('192.0.2.10', main=5)
