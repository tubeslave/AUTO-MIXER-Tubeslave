import time

from backend.ableton_client import AbletonClient, _ableton_vol_to_db
from pythonosc.osc_message_builder import OscMessageBuilder


class DummySocket:
    def __init__(self):
        self.sent = []

    def sendto(self, dgram, target):
        self.sent.append((dgram, target))
        return len(dgram)


class HandshakeSocket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def recvfrom(self, _size):
        return self.replies.pop(0)


def _osc_datagram(address: str) -> bytes:
    return OscMessageBuilder(address=address).build().dgram


def _connected_client() -> AbletonClient:
    client = AbletonClient(ip="127.0.0.1", send_port=11000, recv_port=11001)
    client.is_connected = True
    client.sock = DummySocket()
    client._allowed_reply_ips = {"127.0.0.1"}
    return client


def test_expected_peer_requires_configured_ip_and_source_port():
    client = _connected_client()

    assert client._is_expected_peer(("127.0.0.1", 11000)) is True
    assert client._is_expected_peer(("127.0.0.2", 11000)) is False
    assert client._is_expected_peer(("127.0.0.1", 9999)) is False


def test_handshake_ignores_racing_packets_until_expected_reply():
    client = _connected_client()
    client.recv_sock = HandshakeSocket(
        [
            (_osc_datagram("/live/test"), ("127.0.0.2", 11000)),
            (b"not-osc", ("127.0.0.1", 11000)),
            (_osc_datagram("/attacker"), ("127.0.0.1", 11000)),
            (_osc_datagram("/live/test"), ("127.0.0.1", 11000)),
        ]
    )

    reply = client._receive_valid_handshake(1.0)

    assert reply.address == "/live/test"
    assert len(client.recv_sock.timeouts) == 4


def test_unsolicited_and_unknown_replies_never_enter_state_or_callbacks():
    client = _connected_client()
    updates = []
    client.subscribe("*", lambda address, *args: updates.append((address, args)))

    client._handle_message("/live/track/get/volume", 0, 0.85)
    for index in range(1000):
        client._handle_message(f"/attacker/{index}", index)

    assert client.state == {}
    assert updates == []
    assert client._pending_replies == {}


def test_only_fresh_correlated_track_reply_becomes_observed_state():
    client = _connected_client()

    assert client._send_query("/live/track/get/volume", 0) is True
    client._handle_message("/live/track/get/volume", 0, 0.85)

    assert client.state == {"/ch/1/fdr": _ableton_vol_to_db(0.85)}
    assert client._pending_replies == {}

    client._handle_message("/live/track/get/volume", 0, 0.5)
    assert client.state == {"/ch/1/fdr": _ableton_vol_to_db(0.85)}


def test_expired_correlated_reply_is_ignored():
    client = _connected_client()
    key = ("/live/master/get/volume", None)
    client._pending_replies[key] = time.monotonic() - 1.0

    client._handle_message("/live/master/get/volume", 0.85)

    assert client.state == {}
    assert client._pending_replies == {}


def test_unsupported_query_fails_closed_without_sending():
    client = _connected_client()

    assert client._send_query("/attacker/query") is False
    assert client.sock.sent == []


def test_pending_reply_table_and_track_identifiers_are_bounded():
    client = _connected_client()
    deadline = time.monotonic() + 1.0
    for track_id in range(client._MAX_PENDING_REPLIES):
        client._pending_replies[("/live/track/get/volume", track_id)] = deadline

    assert client._send_query("/live/track/get/mute", 0) is False
    assert client._send_query("/live/track/get/volume", client._MAX_TRACK_ID + 1) is False
    assert len(client._pending_replies) == client._MAX_PENDING_REPLIES
    assert client.sock.sent == []


def test_setters_propagate_transport_failure_without_publishing_requested_state():
    client = _connected_client()
    observed = {
        "/ch/1/fdr": -12.0,
        "/ch/1/mute": 0,
        "/ch/1/pan": 10.0,
        "/ch/1/name": "Observed",
        "/main/1/fdr": -6.0,
    }
    client.state.update(observed)
    client._send_osc = lambda *args, **kwargs: False

    assert client.set_channel_fader(1, 2.0) is False
    assert client.set_channel_mute(1, 1) is False
    assert client.set_channel_pan(1, -50.0) is False
    assert client.set_channel_name(1, "Requested") is False
    assert client.set_main_fader(1, 3.0) is False
    assert client.state == observed


def test_compatibility_send_returns_nested_result_and_does_not_mutate_state():
    client = _connected_client()
    client.state["/ch/1/fdr"] = -9.0
    client.set_channel_fader = lambda channel, value: False

    assert client.send("/ch/1/fdr", 0.0) is False
    assert client.state == {"/ch/1/fdr": -9.0}


def test_successful_submission_is_not_promoted_to_observed_state():
    client = _connected_client()

    assert client.set_channel_fader(1, 0.0) is True
    assert client.state == {}
    assert len(client.sock.sent) == 1
