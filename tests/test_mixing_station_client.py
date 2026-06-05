from pythonosc.osc_message import OscMessage

from backend.mixing_station_client import MixingStationClient


class DummySocket:
    def __init__(self):
        self.sent = []

    def sendto(self, dgram, target):
        self.sent.append((dgram, target))
        return len(dgram)


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


def test_mixing_station_sends_canonical_trim_as_preamp_gain_without_network():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client.is_connected = True
    client.sock = DummySocket()

    result = client.set_channel_gain(2, 1.5)

    assert result is True
    assert len(client.sock.sent) == 1
    message = OscMessage(client.sock.sent[0][0])
    assert message.address == "/con/v/ch.1.preamp.gain"
    assert list(message.params) == [1.5]
    assert client.sock.sent[0][1] == ("127.0.0.1", 9000)


def test_mixing_station_query_parameter_uses_canonical_trim_alias_without_network():
    client = MixingStationClient(host="127.0.0.1", osc_port=9000)
    client.is_connected = True
    client.sock = DummySocket()
    client.state["/ch/3/in/set/trim"] = -2.0

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
