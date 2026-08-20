import socket
from unittest.mock import MagicMock, patch

from pythonosc.osc_message import OscMessage
from pythonosc.osc_message_builder import OscMessageBuilder

from backend.mixing_station_client import MixingStationClient


class DummySocket:
    def __init__(self, on_send=None):
        self.sent = []
        self.on_send = on_send

    def sendto(self, dgram, target):
        self.sent.append((dgram, target))
        if self.on_send:
            self.on_send()
        return len(dgram)


def _osc_datagram(address, value):
    builder = OscMessageBuilder(address=address)
    builder.add_arg(value)
    return builder.build().dgram


def test_mixing_station_maps_preamp_gain_to_canonical_trim_cache_key():
    client = MixingStationClient()
    callbacks = []
    wildcard = []
    client.subscribe("/ch/2/in/set/trim", lambda address, *args: callbacks.append((address, args)))
    client.subscribe("*", lambda address, *args: wildcard.append((address, args)))

    client._handle_message("/con/v/ch.1.preamp.gain", 2.25)

    assert client.state == {"/ch/2/in/set/trim": 2.25}
    assert client.state_timestamps["/ch/2/in/set/trim"] > 0
    assert "/ch/2/gain" not in client.state
    assert callbacks == [("/ch/2/in/set/trim", (2.25,))]
    assert wildcard == [("/ch/2/in/set/trim", (2.25,))]


def test_mixing_station_blocks_canonical_trim_without_authenticated_ipc():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client.is_connected = True
    client.sock = DummySocket()

    result = client.set_channel_gain(2, 1.5)

    assert result is False
    assert client.sock.sent == []


def test_mixing_station_query_parameter_uses_canonical_trim_alias_without_network():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client.is_connected = True
    client.sock = DummySocket(
        on_send=lambda: client._handle_message(
            "/con/v/ch.2.preamp.gain",
            -2.0,
        )
    )

    result = client.query_parameter("/ch/3/gain")

    assert result == {
        "status": "received",
        "address": "/ch/3/in/set/trim",
        "value": -2.0,
        "source": "mixing_station_query",
    }
    message = OscMessage(client.sock.sent[0][0])
    assert message.address == "/con/v/ch.2.preamp.gain"
    assert list(message.params) == []


def test_mixing_station_query_rejects_stale_cached_state_without_fresh_readback():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client.is_connected = True
    client.sock = DummySocket()
    client.state["/ch/3/in/set/trim"] = 12.0
    client.state_timestamps["/ch/3/in/set/trim"] = 1.0

    result = client.query_parameter("/ch/3/gain")

    assert result == {
        "status": "readback_unavailable",
        "address": "/ch/3/in/set/trim",
        "value": None,
        "source": "mixing_station_query",
    }


def test_mixing_station_state_rejects_spoofed_udp_source():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client._expected_osc_sources = frozenset({("127.0.0.1", 9000)})
    packet = _osc_datagram("/con/v/ch.0.mix.lvl", -8.0)

    assert client._process_received_datagram(packet, ("127.0.0.2", 9000)) is False
    assert client.state == {}
    assert client._process_received_datagram(packet, ("127.0.0.1", 9000)) is True
    assert client.state == {"/ch/1/fdr": -8.0}


class ConnectionSocket:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.sent = []
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return ("0.0.0.0", 49152)

    def sendto(self, dgram, target):
        self.sent.append((dgram, target))
        return len(dgram)

    def recvfrom(self, size):
        del size
        item = next(self.responses, socket.timeout())
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.closed = True


def _connect_with_socket(client, fake_socket):
    rest_response = MagicMock(status_code=503)
    with patch(
        "backend.mixing_station_client.socket.getaddrinfo",
        return_value=[
            (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("127.0.0.1", 9000))
        ],
    ), patch(
        "backend.mixing_station_client.socket.socket",
        return_value=fake_socket,
    ), patch(
        "backend.mixing_station_client.requests.get",
        return_value=rest_response,
    ), patch(
        "backend.mixing_station_client.threading.Thread"
    ), patch.object(
        client, "_start_subscription_renewal"
    ), patch.object(
        client, "_scan_mixer_state"
    ):
        return client.connect(timeout=0.05)


def test_mixing_station_connection_rejects_spoofed_and_unrelated_udp_replies():
    fake_socket = ConnectionSocket(
        [
            (_osc_datagram("/con/v/ch.0.mix.lvl", -1.0), ("127.0.0.2", 9000)),
            (_osc_datagram("/con/v/ch.1.mix.lvl", -2.0), ("127.0.0.1", 9000)),
            socket.timeout(),
        ]
    )
    client = MixingStationClient(host="mixing-station.test", osc_port=9000)

    assert _connect_with_socket(client, fake_socket) is False
    assert client.is_connected is False
    assert client.state == {}
    assert fake_socket.closed is False
    assert fake_socket.sent == []


def test_mixing_station_connection_rejects_even_configured_peer_without_authentication():
    fake_socket = ConnectionSocket(
        [
            (_osc_datagram("/con/v/ch.0.mix.lvl", -3.0), ("127.0.0.2", 9000)),
            (_osc_datagram("/con/v/ch.0.mix.lvl", -9.0), ("127.0.0.1", 9000)),
        ]
    )
    client = MixingStationClient(host="mixing-station.test", osc_port=9000)

    assert _connect_with_socket(client, fake_socket) is False
    assert client.state == {}
    assert fake_socket.sent == []
