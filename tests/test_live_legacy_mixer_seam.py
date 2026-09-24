"""Migration tests for handing a canonical mixer session to legacy code."""

import os
import sys
from dataclasses import dataclass

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND)

from live_runtime.legacy_mixer_seam import (  # noqa: E402
    LegacyExternalMixerSeam,
    LegacyMixerSeamError,
)


class FakeClient:
    def __init__(self):
        self.is_connected = True
        self.ip = "10.0.0.2"
        self.port = 2223
        self.tls = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.read_calls = 0
        self.write_calls = 0

    def connect(self):
        self.connect_calls += 1
        return True

    def disconnect(self):
        self.disconnect_calls += 1

    def get_channel_name(self, channel):
        self.read_calls += 1
        return f"CH {channel}"

    def query_route(self, slot):
        self.read_calls += 1
        return f"MAIN {slot}"

    def read_meter(self, output):
        self.read_calls += 1
        return -18.0

    def set_fader(self, channel, value):
        self.write_calls += 1

    def send(self, payload):
        self.write_calls += 1


@dataclass(frozen=True)
class FakeTarget:
    mixer_type: str = "wing"
    ip: str = "10.0.0.2"
    port: int = 2223


class FakeSession:
    def __init__(self):
        self.connected = True
        self.client = FakeClient()
        self.target = FakeTarget()


class FakeLegacyEngine:
    def __init__(self):
        self.mixer_client = None
        self._real_mixer_client = None
        self.mixer_type = "legacy"
        self.mixer_ip = "127.0.0.1"
        self.mixer_port = 9999
        self.discover_calls = 0
        self.connect_calls = 0

    def _discover_mixer(self):
        self.discover_calls += 1
        return False

    def _connect_mixer(self):
        self.connect_calls += 1
        return False


def test_bind_reuses_live_owned_client_without_legacy_discovery_or_connect():
    session = FakeSession()
    engine = FakeLegacyEngine()
    audits = []
    seam = LegacyExternalMixerSeam(engine, session, audit_sink=audits.append)

    assert seam.bind() is True
    assert seam.bind() is False
    assert seam.status().bound is True
    assert seam.status().engine_uses_proxy is True
    assert seam.status().physical_client_id == id(session.client)

    assert engine._discover_mixer() is True
    assert engine._connect_mixer() is True
    assert engine.discover_calls == 0
    assert engine.connect_calls == 0
    assert session.client.connect_calls == 0
    assert session.client.disconnect_calls == 0
    assert engine.mixer_type == "wing"
    assert engine.mixer_ip == "10.0.0.2"
    assert engine.mixer_port == 2223

    events = [event["event"] for event in audits]
    assert events[:3] == [
        "legacy_external_mixer_bound",
        "legacy_external_mixer_discovery_bypassed",
        "legacy_external_mixer_connection_bypassed",
    ]


def test_read_only_proxy_delegates_reads_and_blocks_writes_and_lifecycle():
    session = FakeSession()
    engine = FakeLegacyEngine()
    seam = LegacyExternalMixerSeam(engine, session)
    seam.bind()

    assert engine.mixer_client.get_channel_name(3) == "CH 3"
    assert engine.mixer_client.query_route(1) == "MAIN 1"
    assert engine.mixer_client.read_meter(1) == -18.0
    assert session.client.read_calls == 3

    with pytest.raises(LegacyMixerSeamError, match="set_fader"):
        engine.mixer_client.set_fader(1, -1.0)
    with pytest.raises(LegacyMixerSeamError, match="send"):
        engine.mixer_client.send("unsafe")
    with pytest.raises(LegacyMixerSeamError, match="connect"):
        engine.mixer_client.connect()
    with pytest.raises(LegacyMixerSeamError, match="disconnect"):
        engine.mixer_client.disconnect()

    assert session.client.connect_calls == 0
    assert session.client.disconnect_calls == 0
    assert session.client.write_calls == 0


def test_detach_restores_legacy_slots_methods_and_metadata_without_disconnect():
    session = FakeSession()
    engine = FakeLegacyEngine()
    seam = LegacyExternalMixerSeam(engine, session)
    seam.bind()

    assert seam.detach() is True
    assert seam.detach() is False
    assert seam.status().bound is False
    assert engine.mixer_client is None
    assert engine._real_mixer_client is None
    assert engine.mixer_type == "legacy"
    assert engine.mixer_ip == "127.0.0.1"
    assert engine.mixer_port == 9999
    assert session.client.disconnect_calls == 0

    assert engine._discover_mixer() is False
    assert engine._connect_mixer() is False
    assert engine.discover_calls == 1
    assert engine.connect_calls == 1


def test_bind_fails_closed_when_legacy_already_owns_mixer_client():
    session = FakeSession()
    engine = FakeLegacyEngine()
    engine.mixer_client = FakeClient()
    seam = LegacyExternalMixerSeam(engine, session)

    with pytest.raises(LegacyMixerSeamError, match="ownership is ambiguous"):
        seam.bind()

    assert seam.bound is False


def test_bind_fails_closed_without_connected_live_owner():
    session = FakeSession()
    session.connected = False
    engine = FakeLegacyEngine()
    seam = LegacyExternalMixerSeam(engine, session)

    with pytest.raises(LegacyMixerSeamError, match="not connected"):
        seam.bind()

    assert seam.bound is False


def test_displaced_proxy_or_changed_live_client_fails_closed():
    session = FakeSession()
    engine = FakeLegacyEngine()
    seam = LegacyExternalMixerSeam(engine, session)
    seam.bind()

    engine.mixer_client = None
    with pytest.raises(LegacyMixerSeamError, match="proxy was displaced"):
        engine._connect_mixer()

    engine.mixer_client = engine._real_mixer_client
    session.client = FakeClient()
    with pytest.raises(LegacyMixerSeamError, match="client changed"):
        engine._discover_mixer()
